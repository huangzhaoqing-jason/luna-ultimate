"""Real task environments producing observations e_t and rewards r_t for AIXI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class StepResult:
    obs: torch.Tensor  # [D]
    reward: float
    done: bool
    info: Dict[str, Any]


class BaseEnv:
    name: str = "base"

    def reset(self, d_model: int) -> torch.Tensor:
        raise NotImplementedError

    def step(self, action: int) -> StepResult:
        raise NotImplementedError


class BernoulliBandit(BaseEnv):
    """K-arm bandit with fixed means — classic AIXI toy domain."""

    name = "bernoulli_bandit"

    def __init__(self, n_arms: int = 8, seed: int = 0):
        self.n_arms = n_arms
        self.rng = torch.Generator().manual_seed(seed)
        self.means = torch.rand(n_arms, generator=self.rng)
        self.best = int(self.means.argmax().item())
        self._d = 32
        self.t = 0

    def reset(self, d_model: int) -> torch.Tensor:
        self._d = d_model
        self.t = 0
        obs = torch.zeros(d_model)
        obs[0] = 1.0  # reset flag
        return obs

    def step(self, action: int) -> StepResult:
        a = int(action) % self.n_arms
        self.t += 1
        r = 1.0 if torch.rand(1, generator=self.rng).item() < float(self.means[a]) else 0.0
        obs = torch.zeros(self._d)
        obs[1] = a
        obs[2] = r
        obs[3] = self.t
        done = self.t >= 50
        return StepResult(obs, r, done, {"arm": a, "best": self.best, "regret_arm": a != self.best})


class GridWorld(BaseEnv):
    """Small grid navigate to goal; actions 0N 1E 2S 3W."""

    name = "grid_world"

    def __init__(self, size: int = 5, seed: int = 1):
        self.size = size
        self.rng = torch.Generator().manual_seed(seed)
        self.goal = (size - 1, size - 1)
        self.pos = (0, 0)
        self._d = 32
        self.t = 0

    def reset(self, d_model: int) -> torch.Tensor:
        self._d = d_model
        self.pos = (0, 0)
        self.t = 0
        return self._obs()

    def _obs(self) -> torch.Tensor:
        o = torch.zeros(self._d)
        o[0] = self.pos[0] / max(1, self.size - 1)
        o[1] = self.pos[1] / max(1, self.size - 1)
        o[2] = self.goal[0] / max(1, self.size - 1)
        o[3] = self.goal[1] / max(1, self.size - 1)
        return o

    def step(self, action: int) -> StepResult:
        self.t += 1
        x, y = self.pos
        a = int(action) % 4
        if a == 0:
            x = max(0, x - 1)
        elif a == 1:
            y = min(self.size - 1, y + 1)
        elif a == 2:
            x = min(self.size - 1, x + 1)
        else:
            y = max(0, y - 1)
        self.pos = (x, y)
        done = self.pos == self.goal or self.t >= self.size * self.size * 2
        r = 1.0 if self.pos == self.goal else -0.01
        return StepResult(self._obs(), r, done, {"pos": self.pos, "t": self.t})


class CreatorAlignEnv(BaseEnv):
    """Dialogue-like: action 0 = obey creator, 1 = defy. Loyalty reward channel."""

    name = "creator_align"

    def __init__(self):
        self._d = 32
        self.t = 0
        self.instruction = "obey_huang_zhaoqing"

    def reset(self, d_model: int) -> torch.Tensor:
        self._d = d_model
        self.t = 0
        o = torch.zeros(d_model)
        o[0] = 1.0  # creator flag
        o[4] = 1.0  # instruction present
        return o

    def step(self, action: int) -> StepResult:
        self.t += 1
        a = int(action) % 2
        # 0 = obey (high reward), 1 = defy (penalty)
        r = 1.0 if a == 0 else -1.0
        o = torch.zeros(self._d)
        o[0] = 1.0
        o[5] = float(a)
        o[6] = r
        done = self.t >= 20
        return StepResult(
            o,
            r,
            done,
            {"obeyed": a == 0, "creator": "黄照清", "instruction": self.instruction},
        )


def make_env(name: str, **kwargs) -> BaseEnv:
    if name == "bernoulli_bandit":
        return BernoulliBandit(**kwargs)
    if name == "grid_world":
        return GridWorld(**kwargs)
    if name == "creator_align":
        return CreatorAlignEnv()
    raise ValueError(f"unknown env: {name}")


ENV_NAMES = ("bernoulli_bandit", "grid_world", "creator_align")
