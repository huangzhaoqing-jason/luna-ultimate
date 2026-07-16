"""小脑：高频任务缓存 + 蒸馏小模型接口。

简单任务由丘脑唤醒小脑路径，命中缓存则跳过重推理。
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from config import LunaConfig


@dataclass
class CacheHit:
    key: str
    value: torch.Tensor
    hits: int


class CerebellarCorrector(nn.Module):
    """轻量纠错头 + LRU 响应缓存。"""

    def __init__(self, config: LunaConfig, cache_size: int = 256):
        super().__init__()
        self.d_model = config.hidden_size
        self.correct = nn.Linear(self.d_model, self.d_model, bias=False)
        self.cache_size = cache_size
        self._cache: "OrderedDict[str, Tuple[torch.Tensor, int]]" = OrderedDict()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """残差纠错：hidden + correct(hidden)。"""
        return hidden + self.correct(hidden)

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]

    def lookup(self, text: str) -> Optional[CacheHit]:
        k = self._key(text or "")
        if k not in self._cache:
            return None
        val, hits = self._cache.pop(k)
        hits += 1
        self._cache[k] = (val, hits)
        return CacheHit(key=k, value=val, hits=hits)

    def store(self, text: str, value: torch.Tensor) -> str:
        k = self._key(text or "")
        self._cache[k] = (value.detach().cpu(), 1)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return k

    def stats(self) -> Dict[str, int]:
        return {"size": len(self._cache), "capacity": self.cache_size}
