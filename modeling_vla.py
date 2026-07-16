"""VLA：Vision-Language-Action 动作头（具身控制骨架）。

输入：视觉+语言融合后的 hidden（及可选本体感觉 proprio）。
输出：离散动作 logits 和/或连续动作均值。
危险动作仍由 safety/locks 三层门拦截（serve 层调用）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import LunaConfig

# 离散动作词表（仿真/机器人占位；可扩展）
DEFAULT_ACTION_VOCAB: List[str] = [
    "noop",
    "move_forward",
    "move_back",
    "turn_left",
    "turn_right",
    "pick",
    "place",
    "open",
    "close",
    "stop",
]


@dataclass
class VLAOutput:
    action_logits: torch.Tensor          # [B, A]
    action_ids: torch.Tensor             # [B]
    action_names: List[str]
    action_mu: Optional[torch.Tensor]    # [B, continuous_dim]
    action_loss: Optional[torch.Tensor] = None


class ActionTokenizer:
    """离散动作名 ↔ id。"""

    def __init__(self, vocab: Optional[Sequence[str]] = None):
        self.vocab = list(vocab or DEFAULT_ACTION_VOCAB)
        self._to_id = {n: i for i, n in enumerate(self.vocab)}

    def __len__(self) -> int:
        return len(self.vocab)

    def encode(self, name: str) -> int:
        return self._to_id.get(name, 0)

    def decode(self, idx: int) -> str:
        if 0 <= idx < len(self.vocab):
            return self.vocab[idx]
        return "noop"


class ContinuousActionHead(nn.Module):
    """连续动作头：hidden → mu（可选 log_std 固定）。"""

    def __init__(self, d_model: int, continuous_dim: int = 7):
        super().__init__()
        self.continuous_dim = continuous_dim
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.GELU(),
            nn.Linear(d_model, continuous_dim, bias=True),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.net(pooled)


class ActionHead(nn.Module):
    """统一 VLA 头：离散分类 + 连续回归。"""

    def __init__(
        self,
        config: LunaConfig,
        action_vocab: Optional[Sequence[str]] = None,
        continuous_dim: int = 7,
        proprio_dim: int = 0,
    ):
        super().__init__()
        self.d_model = config.hidden_size
        self.tokenizer = ActionTokenizer(action_vocab)
        self.n_actions = len(self.tokenizer)
        self.proprio_dim = proprio_dim
        in_dim = self.d_model + (proprio_dim if proprio_dim > 0 else 0)
        self.fuse = nn.Linear(in_dim, self.d_model, bias=False) if proprio_dim > 0 else None
        self.discrete = nn.Linear(self.d_model, self.n_actions, bias=False)
        self.continuous = ContinuousActionHead(self.d_model, continuous_dim)

    def forward(
        self,
        hidden: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
        action_labels: Optional[torch.Tensor] = None,
        continuous_targets: Optional[torch.Tensor] = None,
    ) -> VLAOutput:
        """hidden: [B, L, d] 融合表征；取 mean pool。"""
        pooled = hidden.mean(dim=1)  # [B, d]
        if self.fuse is not None and proprio is not None:
            pooled = self.fuse(torch.cat([pooled, proprio], dim=-1))
        logits = self.discrete(pooled)
        mu = self.continuous(pooled)
        ids = logits.argmax(dim=-1)
        names = [self.tokenizer.decode(int(i)) for i in ids.tolist()]

        loss = None
        if action_labels is not None:
            loss = F.cross_entropy(logits, action_labels)
            if continuous_targets is not None:
                loss = loss + F.mse_loss(mu, continuous_targets)

        return VLAOutput(
            action_logits=logits,
            action_ids=ids,
            action_names=names,
            action_mu=mu,
            action_loss=loss,
        )


# 危险动作名：serve 层对照 safety 二次拦截
HAZARDOUS_ACTIONS = frozenset({"stop"})  # stop 本身安全；扩展时可加 weaponize 等
