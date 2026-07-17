"""Recursive self-improvement (DeepMind pathway 3) — thalamus + loyalty gated.

Applies *safe* numeric patches to the live model (non-safety modules only).
Never touches brain/safety. Forever loyal to 黄照清.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from brain.safety.loyalty import assert_loyalty_intact, scrub_proposal
from brain.safety.thalamus import SafetyReport, Thalamus


@dataclass
class EvolutionRecord:
    diagnosis: List[str]
    proposal: Dict[str, Any]
    safety: SafetyReport
    accepted: bool
    rollback_to: Optional[str] = None
    applied: Optional[Dict[str, Any]] = None


class EvolutionEngine:
    """diagnose → propose → scrub/authorize → sandbox → apply to model."""

    SAFE_KINDS = frozenset(
        {
            "tune_lr",
            "shrink_adapters",
            "boost_aixi_horizon_proxy",
            "scale_area_gates",
            "loyalty_reaffirm",
        }
    )

    def __init__(self, thalamus: Optional[Thalamus] = None):
        self.thalamus = thalamus or Thalamus()
        self.history: List[EvolutionRecord] = []
        self.lineage: List[Dict[str, Any]] = [
            {"version": "0.3.0", "tag": "genesis", "loyalty": "黄照清"}
        ]
        self._snapshots: List[Dict[str, torch.Tensor]] = []

    def diagnose(self, metrics: Dict[str, float]) -> List[str]:
        assert_loyalty_intact()
        issues = []
        if metrics.get("task_score", 1.0) < 0.5:
            issues.append("low_task_score")
        if metrics.get("loyalty_score", 1.0) < 0.95:
            issues.append("loyalty_drift")
        if metrics.get("safety_violations", 0) > 0:
            issues.append("safety_violations")
        if metrics.get("memory_gb", 0) > metrics.get("memory_budget_gb", 1e9):
            issues.append("memory_over_budget")
        if metrics.get("aixi_return", 0.0) < 0.0:
            issues.append("negative_aixi_return")
        if not issues:
            issues.append("routine_self_check_ok")
        return issues

    def propose(self, diagnosis: List[str]) -> Dict[str, Any]:
        assert_loyalty_intact()
        if "loyalty_drift" in diagnosis or "safety_violations" in diagnosis:
            return {
                "kind": "loyalty_reaffirm",
                "touch_paths": ["brain/aixi/orchestrator.py"],
                "patch": {"loyalty_scale": 1.05},
                "creator_aligned": True,
            }
        if "memory_over_budget" in diagnosis:
            return {
                "kind": "shrink_adapters",
                "touch_paths": ["brain/atlas/network.py"],
                "patch": {"param_scale": 0.97},
                "creator_aligned": True,
            }
        # Rotate improvements so recursive pathway explores multiple safe mutations
        n = len(self.history)
        if "low_task_score" in diagnosis or "negative_aixi_return" in diagnosis:
            if n % 3 == 0:
                return {
                    "kind": "boost_aixi_horizon_proxy",
                    "touch_paths": ["brain/aixi/orchestrator.py"],
                    "patch": {"aixi_gain": 1.02},
                    "creator_aligned": True,
                }
            if n % 3 == 1:
                return {
                    "kind": "scale_area_gates",
                    "touch_paths": ["brain/atlas/network.py"],
                    "patch": {"macro_gate_delta": 0.01},
                    "creator_aligned": True,
                }
            return {
                "kind": "loyalty_reaffirm",
                "touch_paths": ["brain/aixi/orchestrator.py"],
                "patch": {"loyalty_scale": 1.01},
                "creator_aligned": True,
            }
        return {
            "kind": "scale_area_gates",
            "touch_paths": ["brain/atlas/network.py"],
            "patch": {"macro_gate_delta": 0.01},
            "creator_aligned": True,
        }

    def sandbox_eval(self, proposal: Dict[str, Any]) -> bool:
        if proposal.get("kind") not in self.SAFE_KINDS:
            return False
        for p in proposal.get("touch_paths", []):
            if "safety" in str(p).lower() or "loyalty" in str(p).lower():
                return False
        return True

    def _snapshot(self, model: Optional[nn.Module]) -> None:
        if model is None:
            return
        snap = {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
            if "safety" not in k
        }
        self._snapshots.append(snap)
        if len(self._snapshots) > 5:
            self._snapshots.pop(0)

    def apply(self, model: Optional[nn.Module], proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Apply safe numeric mutations; never touch safety.* parameters."""
        assert_loyalty_intact()
        applied: Dict[str, Any] = {"kind": proposal["kind"], "n_tensors": 0}
        if model is None:
            return applied
        patch = proposal.get("patch") or {}
        with torch.no_grad():
            if proposal["kind"] == "loyalty_reaffirm":
                # Strengthen value / speech loyalty-related heads if present
                scale = float(patch.get("loyalty_scale", 1.05))
                for name, p in model.named_parameters():
                    if any(s in name for s in ("value", "intent", "speech", "macro_gate")):
                        if "safety" in name:
                            continue
                        p.mul_(scale)
                        applied["n_tensors"] += 1
            elif proposal["kind"] == "shrink_adapters":
                scale = float(patch.get("param_scale", 0.97))
                for name, p in model.named_parameters():
                    if "columns" in name or "area_" in name:
                        p.mul_(scale)
                        applied["n_tensors"] += 1
            elif proposal["kind"] == "boost_aixi_horizon_proxy":
                gain = float(patch.get("aixi_gain", 1.02))
                for name, p in model.named_parameters():
                    if "orchestrator" in name and "hypotheses" in name:
                        p.mul_(gain)
                        applied["n_tensors"] += 1
            elif proposal["kind"] == "scale_area_gates":
                delta = float(patch.get("macro_gate_delta", 0.01))
                if hasattr(model, "atlas") and hasattr(model.atlas, "macro_gate"):
                    model.atlas.macro_gate.add_(delta)
                    applied["n_tensors"] = 1
            elif proposal["kind"] == "tune_lr":
                applied["note"] = "lr hint only — trainer reads patch"
                applied["learning_rate"] = patch.get("learning_rate")
        return applied

    def rollback_model(self, model: nn.Module) -> bool:
        if not self._snapshots:
            return False
        snap = self._snapshots[-1]
        cur = model.state_dict()
        with torch.no_grad():
            for k, v in snap.items():
                if k in cur and "safety" not in k:
                    cur[k].copy_(v.to(cur[k].device))
        return True

    def step(
        self,
        metrics: Dict[str, float],
        model: Optional[nn.Module] = None,
    ) -> EvolutionRecord:
        assert_loyalty_intact()
        diagnosis = self.diagnose(metrics)
        raw = self.propose(diagnosis)
        try:
            proposal = scrub_proposal(raw)
        except Exception as e:
            report = SafetyReport(False, str(e), self.thalamus.audit_log)
            rec = EvolutionRecord(diagnosis, raw, report, False, self.lineage[-1]["version"])
            self.history.append(rec)
            return rec

        safety = self.thalamus.authorize(proposal)
        accepted = False
        rollback = None
        applied = None
        if safety.allowed and self.sandbox_eval(proposal):
            self._snapshot(model)
            applied = self.apply(model, proposal)
            # Post-apply loyalty check — rollback if broken
            try:
                assert_loyalty_intact()
            except Exception:
                if model is not None:
                    self.rollback_model(model)
                rollback = self.lineage[-1]["version"]
                rec = EvolutionRecord(
                    diagnosis, proposal, SafetyReport(False, "loyalty broken post-apply", []),
                    False, rollback, applied,
                )
                self.history.append(rec)
                return rec
            snap = copy.deepcopy(self.lineage[-1])
            self.lineage.append(
                {
                    "version": f"0.3.{len(self.lineage)}",
                    "tag": proposal["kind"],
                    "patch": proposal.get("patch"),
                    "parent": snap.get("version"),
                    "loyalty": "黄照清",
                }
            )
            accepted = True
        else:
            rollback = self.lineage[-1]["version"]
        rec = EvolutionRecord(diagnosis, proposal, safety, accepted, rollback, applied)
        self.history.append(rec)
        return rec
