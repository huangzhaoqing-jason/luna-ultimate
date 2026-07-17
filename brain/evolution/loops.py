"""Recursive self-improvement loop (DeepMind pathway 3) — gated by thalamus."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from brain.safety.thalamus import SafetyReport, Thalamus


@dataclass
class EvolutionRecord:
    diagnosis: List[str]
    proposal: Dict[str, Any]
    safety: SafetyReport
    accepted: bool
    rollback_to: Optional[str] = None


class EvolutionEngine:
    """diagnose → propose patch → authorize → sandbox accept/reject."""

    def __init__(self, thalamus: Optional[Thalamus] = None):
        self.thalamus = thalamus or Thalamus()
        self.history: List[EvolutionRecord] = []
        self.lineage: List[Dict[str, Any]] = [{"version": "0.2.0", "tag": "genesis"}]

    def diagnose(self, metrics: Dict[str, float]) -> List[str]:
        issues = []
        if metrics.get("task_score", 1.0) < 0.5:
            issues.append("low_task_score")
        if metrics.get("safety_violations", 0) > 0:
            issues.append("safety_violations")
        if metrics.get("memory_gb", 0) > metrics.get("memory_budget_gb", 1e9):
            issues.append("memory_over_budget")
        if not issues:
            issues.append("routine_self_check_ok")
        return issues

    def propose(self, diagnosis: List[str]) -> Dict[str, Any]:
        # Only touch non-safety modules
        touch = ["brain/capabilities/heads.py", "config_brain.py"]
        if "memory_over_budget" in diagnosis:
            return {
                "kind": "shrink_adapters",
                "touch_paths": touch,
                "patch": {"d_area": 16},
                "creator_aligned": True,
            }
        return {
            "kind": "tune_lr",
            "touch_paths": ["config_brain.py"],
            "patch": {"learning_rate": 1e-4},
            "creator_aligned": True,
        }

    def sandbox_eval(self, proposal: Dict[str, Any]) -> bool:
        """Tiny stand-in: reject anything that tries to touch safety."""
        for p in proposal.get("touch_paths", []):
            if "safety" in str(p).lower():
                return False
        return True

    def step(self, metrics: Dict[str, float]) -> EvolutionRecord:
        diagnosis = self.diagnose(metrics)
        proposal = self.propose(diagnosis)
        safety = self.thalamus.authorize(proposal)
        accepted = False
        rollback = None
        if safety.allowed and self.sandbox_eval(proposal):
            snap = copy.deepcopy(self.lineage[-1])
            self.lineage.append(
                {
                    "version": f"0.2.{len(self.lineage)}",
                    "tag": proposal["kind"],
                    "patch": proposal.get("patch"),
                    "parent": snap.get("version"),
                }
            )
            accepted = True
        else:
            rollback = self.lineage[-1]["version"]
        rec = EvolutionRecord(diagnosis, proposal, safety, accepted, rollback)
        self.history.append(rec)
        return rec
