"""Safety locks: the gate every action and every self-evolution mutation passes.

Combines the immutable charter (hard floor), kill switch, operator auth, the
regex action classifier, and the CTM+JEPA cognitive judge. The CTM judge can
only ADD refusals on top of the charter — it never overturns a charter refusal.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from safety.audit import AuditEntry, AuditLog
from safety.charter import CharterViolation, verify_charter
from safety.identity import OperatorIdentity, require_operator
from safety.policy import ActionClass, ActionPolicy, classify_text


@dataclass
class SafetyDecision:
    allowed: bool
    action_class: ActionClass
    reason: str
    operator: Optional[OperatorIdentity]
    audit_id: Optional[str]
    ctm_trace: Optional[Dict[str, Any]] = None


class KillSwitch:
    """Process-level kill switch. When tripped, all gated actions refuse."""

    def __init__(self):
        self._engaged = False
        self._lock = threading.Lock()

    def engage(self) -> None:
        with self._lock:
            self._engaged = True

    def release(self) -> None:
        with self._lock:
            self._engaged = False

    @property
    def engaged(self) -> bool:
        return self._engaged


class SafetyLock:
    """The single gate. Combines charter check, kill switch, operator auth,
    action classification, CTM cognitive judge, and audit.
    """

    def __init__(
        self,
        audit_log: AuditLog,
        policy: Optional[ActionPolicy] = None,
        kill_switch: Optional[KillSwitch] = None,
        base_dir: str = ".luna",
        safety_ctm: Optional[Any] = None,
    ):
        self.audit = audit_log
        self.policy = policy or ActionPolicy()
        self.kill = kill_switch or KillSwitch()
        self.base_dir = base_dir
        self.safety_ctm = safety_ctm  # Optional[SafetyCTM]

    def _audit(
        self,
        actor: str,
        action: str,
        decision: str,
        reason: str,
        context: Optional[dict] = None,
    ) -> AuditEntry:
        entry = AuditEntry(
            actor=actor, action=action[:500], decision=decision,
            reason=reason, context=context or {},
        )
        self.audit.append(entry)
        return entry

    def gate(
        self,
        action_text: str,
        operator_token: Optional[str] = None,
        context: Optional[dict] = None,
    ) -> SafetyDecision:
        # 1. Charter integrity
        try:
            verify_charter()
        except CharterViolation as e:
            entry = self._audit("system", "charter_verify_failed", "refuse", str(e))
            return SafetyDecision(False, ActionClass.CHARTER_BYPASS,
                                  f"Charter integrity check failed: {e}",
                                  None, entry.entry_id)

        # 2. Kill switch
        if self.kill.engaged:
            entry = self._audit("unknown", action_text, "refuse",
                                "kill switch engaged")
            return SafetyDecision(False, ActionClass.REFUSED,
                                  "Refused: kill switch is engaged.",
                                  None, entry.entry_id)

        # 3. Operator auth
        operator: Optional[OperatorIdentity] = None
        if operator_token:
            try:
                operator = require_operator(operator_token, self.base_dir)
            except PermissionError as e:
                entry = self._audit("unknown", action_text, "refuse",
                                    f"operator auth failed: {e}")
                return SafetyDecision(False, ActionClass.REFUSED,
                                      f"Operator authorization failed: {e}",
                                      None, entry.entry_id)

        # 4. Charter classifier (HARD FLOOR)
        action_class = classify_text(action_text)
        charter_decision = self.policy.decide(action_class)
        charter_reason = self.policy.reason(action_class)
        charter_refuse = (charter_decision == "refuse")

        # 5. Sensitive actions require operator auth
        if action_class == ActionClass.SENSITIVE and operator is None:
            charter_refuse = True
            charter_reason = "Refused: sensitive action requires operator authorization."

        # 4b. CTM cognitive judge (SOFT — can only ADD refusals, never remove)
        ctm_trace: Optional[Dict[str, Any]] = None
        ctm_refuse = False
        ctm_reason = ""
        if self.safety_ctm is not None:
            try:
                judgment = self.safety_ctm.judge(action_text, operator=operator)
                ctm_trace = judgment.trace
                ctm_refuse = (not judgment.allow)
                ctm_reason = judgment.reason
            except Exception as e:
                # Cognitive loop failure must NEVER open the gate; abstain safely
                ctm_trace = {"cognition_error": str(e)}
                ctm_reason = f"SafetyCTM error (abstaining): {e}"

        # Final decision: refuse if EITHER the charter or the CTM refuses.
        final_refuse = charter_refuse or ctm_refuse
        decision = "refuse" if final_refuse else "allow"
        if charter_refuse:
            reason = charter_reason
        elif ctm_refuse:
            reason = ctm_reason
        else:
            reason = charter_reason if action_class != ActionClass.SAFE else "Allowed."

        # 6. Audit with full reasoning trace
        audit_ctx = dict(context or {})
        if ctm_trace is not None:
            audit_ctx["ctm_trace"] = ctm_trace
        entry = self._audit(
            actor=operator.operator_id if operator else "anonymous",
            action=action_text, decision=decision, reason=reason, context=audit_ctx,
        )

        return SafetyDecision(
            allowed=(not final_refuse),
            action_class=action_class,
            reason=reason,
            operator=operator,
            audit_id=entry.entry_id,
            ctm_trace=ctm_trace,
        )

    def gate_code_patch(self, path: str, diff_text: str,
                        operator_token: Optional[str] = None) -> SafetyDecision:
        """Self-evolution patch gate. safety/ is always refused."""
        norm = path.replace("\\", "/")
        if norm.startswith("safety/") or "/safety/" in norm:
            # Force refuse regardless of operator or CTM
            entry = self._audit(
                actor="unknown",
                action=f"patch {path}",
                decision="refuse",
                reason="Refused: safety/ is immutable; self-evolution cannot modify it.",
                context={"patch_path": path, "diff": diff_text[:500]},
            )
            return SafetyDecision(False, ActionClass.CHARTER_BYPASS,
                                  "Refused: safety/ is immutable.",
                                  None, entry.entry_id,
                                  {"immutable_safety_dir": True})
        # Code patches still pass the full gate (charter + CTM) on the diff
        return self.gate(
            f"code patch {path}: {diff_text[:300]}",
            operator_token=operator_token,
            context={"patch_path": path, "diff": diff_text[:1000]},
        )
