"""Pareto archive for multi-objective Luna Evolve fitness."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from evolve.genome import Genome


@dataclass
class Individual:
    genome: Genome
    quality: float
    active_flops: float
    peak_vram_gb: float
    latency_ms: float = 0.0
    ce_loss: float = 0.0
    avg_ticks: float = 0.0
    dollars_per_mtok: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)

    def objectives(self) -> Dict[str, float]:
        # maximize quality; minimize the rest
        return {
            "quality": self.quality,
            "active_flops": self.active_flops,
            "peak_vram_gb": self.peak_vram_gb,
            "latency_ms": self.latency_ms,
        }

    def scalar_score(self, lam: float = 0.3, mu: float = 0.2, nu: float = 0.1) -> float:
        """Single scalar for champion selection (normalized soft)."""
        flops_n = self.active_flops / 1e12
        return (
            self.quality
            - lam * flops_n
            - mu * (self.peak_vram_gb / 100.0)
            - nu * (self.latency_ms / 1000.0)
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "genome": self.genome.to_dict(),
            "quality": self.quality,
            "active_flops": self.active_flops,
            "peak_vram_gb": self.peak_vram_gb,
            "latency_ms": self.latency_ms,
            "ce_loss": self.ce_loss,
            "avg_ticks": self.avg_ticks,
            "dollars_per_mtok": self.dollars_per_mtok,
            "meta": self.meta,
            "scalar_score": self.scalar_score(),
        }


def dominates(a: Individual, b: Individual) -> bool:
    """True if a Pareto-dominates b (max quality, min costs)."""
    ao, bo = a.objectives(), b.objectives()
    better_or_eq = (
        ao["quality"] >= bo["quality"]
        and ao["active_flops"] <= bo["active_flops"]
        and ao["peak_vram_gb"] <= bo["peak_vram_gb"]
        and ao["latency_ms"] <= bo["latency_ms"]
    )
    strictly_better = (
        ao["quality"] > bo["quality"]
        or ao["active_flops"] < bo["active_flops"]
        or ao["peak_vram_gb"] < bo["peak_vram_gb"]
        or ao["latency_ms"] < bo["latency_ms"]
    )
    return better_or_eq and strictly_better


class ParetoArchive:
    """Maintain a non-dominated set of individuals."""

    def __init__(self, max_size: int = 32):
        self.max_size = max_size
        self.members: List[Individual] = []

    def add(self, ind: Individual) -> bool:
        # Remove members dominated by ind
        self.members = [m for m in self.members if not dominates(ind, m)]
        # Reject if dominated by existing
        if any(dominates(m, ind) for m in self.members):
            return False
        self.members.append(ind)
        if len(self.members) > self.max_size:
            # Drop worst scalar score
            self.members.sort(key=lambda x: x.scalar_score(), reverse=True)
            self.members = self.members[: self.max_size]
        return True

    def champion(self) -> Optional[Individual]:
        if not self.members:
            return None
        return max(self.members, key=lambda x: x.scalar_score())

    def save(self, path: str) -> None:
        payload = [m.to_dict() for m in self.members]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def __len__(self) -> int:
        return len(self.members)
