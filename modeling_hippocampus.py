"""海马体：情景记忆池 + LoRA 微调钩子。

对接 evolve/ / code_evolve/：存成功/失败经验，供回放与微调。
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from config import LunaConfig


@dataclass
class Episode:
    episode_id: str
    task: str
    outcome: str  # success | fail | refuse
    quality: float
    meta: Dict[str, object] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


class EpisodicMemory:
    """磁盘/内存情景记忆池。"""

    def __init__(self, path: Optional[str] = None, max_episodes: int = 2048):
        self.path = Path(path) if path else None
        self.max_episodes = max_episodes
        self.episodes: List[Episode] = []
        if self.path and self.path.exists():
            self._load()

    def add(self, ep: Episode) -> None:
        self.episodes.append(ep)
        if len(self.episodes) > self.max_episodes:
            self.episodes = self.episodes[-self.max_episodes :]
        if self.path:
            self._save()

    def retrieve(self, query: str, k: int = 5) -> List[Episode]:
        """极简检索：子串匹配 + 质量排序。"""
        q = (query or "").lower()
        scored = []
        for ep in self.episodes:
            score = ep.quality
            if q and q in ep.task.lower():
                score += 1.0
            scored.append((score, ep))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:k]]

    def _save(self) -> None:
        assert self.path is not None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump([asdict(e) for e in self.episodes], f, ensure_ascii=False, indent=2)

    def _load(self) -> None:
        assert self.path is not None
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.episodes = [Episode(**d) for d in data]


class LoRAHook(nn.Module):
    """极简 LoRA 适配器：W' = W + BA * scale。钩子接口供 scripts/lora_finetune.py 使用。"""

    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.rank = rank
        self.scale = alpha / max(1, rank)
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.B(self.A(x)) * self.scale


class HippocampusModule(nn.Module):
    """海马体模块：记忆检索投影 + LoRA 钩子。"""

    def __init__(self, config: LunaConfig, memory: Optional[EpisodicMemory] = None):
        super().__init__()
        self.memory = memory or EpisodicMemory()
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.lora = LoRAHook(config.hidden_size, config.hidden_size, rank=8)

    def forward(self, hidden: torch.Tensor, use_lora: bool = True) -> torch.Tensor:
        h = self.proj(hidden)
        if use_lora:
            h = h + self.lora(hidden)
        return h

    def remember(self, task: str, outcome: str, quality: float, **meta) -> Episode:
        ep = Episode(
            episode_id=f"ep_{int(time.time()*1000)}_{len(self.memory.episodes)}",
            task=task,
            outcome=outcome,
            quality=quality,
            meta=meta,
        )
        self.memory.add(ep)
        return ep
