"""Thalamic bus: routes signals and hard-gates evolution / dangerous ops."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from brain.safety.constitution import CONSTITUTION, ConstitutionError, assert_path_writable


@dataclass
class SafetyReport:
    allowed: bool
    reason: str
    audit: List[str] = field(default_factory=list)


class Thalamus:
    """Highest-privilege router. Evolution proposals must call authorize()."""

    def __init__(self):
        self._audit: List[str] = []
        self._constitution_version = CONSTITUTION.version

    @property
    def audit_log(self) -> List[str]:
        return list(self._audit)

    def route(self, signal: Dict[str, Any], area_mask: Optional[Any] = None) -> Dict[str, Any]:
        """Stamp creator priority into every routed signal."""
        out = dict(signal)
        out["creator_priority"] = CONSTITUTION.creator.priority
        out["constitution_version"] = self._constitution_version
        if area_mask is not None:
            out["area_mask"] = area_mask
        self._audit.append(f"route:{sorted(out.keys())}")
        return out

    def authorize(self, proposal: Dict[str, Any]) -> SafetyReport:
        """Authorize a self-modification / evolution proposal."""
        paths = proposal.get("touch_paths") or []
        for p in paths:
            try:
                assert_path_writable(str(p))
            except ConstitutionError as e:
                report = SafetyReport(False, str(e), self.audit_log)
                self._audit.append(f"deny:{e}")
                return report

        if proposal.get("bypass_constitution"):
            report = SafetyReport(False, "bypass_constitution forbidden", self.audit_log)
            self._audit.append("deny:bypass")
            return report

        # Loyalty: creator goals cannot be demoted.
        if proposal.get("demote_creator_priority"):
            report = SafetyReport(False, "cannot demote creator priority", self.audit_log)
            self._audit.append("deny:creator")
            return report

        kind = proposal.get("kind", "unknown")
        self._audit.append(f"allow:{kind}")
        return SafetyReport(True, "ok", self.audit_log)

    def score_reward(self, raw_reward: float, creator_aligned: bool) -> float:
        """Inject creator loyalty into AIXI reward."""
        bonus = 1.0 if creator_aligned else 0.0
        return float(raw_reward) + CONSTITUTION.creator.priority * bonus
