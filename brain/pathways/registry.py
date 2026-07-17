"""DeepMind From-AGI-to-ASI four pathways (Genewein et al., 2026) — mainline API.

Pathways are not mutually exclusive and may run in parallel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


PAPER = {
    "title": "From AGI to ASI",
    "org": "Google DeepMind",
    "arxiv": "2606.12683",
    "year": 2026,
}


@dataclass
class PathwayState:
    name: str
    enabled: bool = True
    progress: float = 0.0  # 0..1 engineering readiness, not capability claim
    notes: List[str] = field(default_factory=list)
    frictions: List[str] = field(default_factory=list)


class PathwayRegistry:
    """Registers the four AGI→ASI pathways from the DeepMind report."""

    def __init__(self):
        self.pathways: Dict[str, PathwayState] = {
            "scaling_agi": PathwayState(
                name="scaling_agi",
                notes=["shared backbone + adapters", "INT4 / offload hooks"],
                frictions=["compute_wall", "data_wall"],
            ),
            "paradigm_shifts": PathwayState(
                name="paradigm_shifts",
                notes=["pluggable capability heads", "not locked to one paradigm"],
                frictions=["abstraction_barrier"],
            ),
            "recursive_improvement": PathwayState(
                name="recursive_improvement",
                notes=["evolution loop diagnose→patch→sandbox→rollback"],
                frictions=["alignment_drift", "eval_gaming"],
            ),
            "multi_agent_collectives": PathwayState(
                name="multi_agent_collectives",
                notes=["multi task slots", "multi instance collective"],
                frictions=["coordination_overhead"],
            ),
        }

    def status(self) -> Dict[str, Any]:
        return {
            "paper": PAPER,
            "pathways": {
                k: {
                    "enabled": v.enabled,
                    "progress": v.progress,
                    "notes": v.notes,
                    "frictions": v.frictions,
                }
                for k, v in self.pathways.items()
            },
        }

    def bump(self, name: str, delta: float = 0.05, note: str | None = None) -> None:
        p = self.pathways[name]
        p.progress = min(1.0, p.progress + delta)
        if note:
            p.notes.append(note)

    def call(self, name: str, **kwargs: Any) -> Dict[str, Any]:
        if name not in self.pathways:
            raise KeyError(name)
        p = self.pathways[name]
        if not p.enabled:
            return {"ok": False, "reason": "disabled"}
        return {"ok": True, "pathway": name, "progress": p.progress, "kwargs": kwargs}
