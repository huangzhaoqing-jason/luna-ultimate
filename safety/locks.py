"""Safety locks: the gate every action and every self-evolution mutation passes."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Optional

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
    action classification, and audit.
    """

    def __init__(
        self,
        audit_log: AuditLog,
        policy: Optional[ActionPolicy] = None,
        kill_switch: Optional[KillSwitch] = None,
        base_dir: str = ".luna",
    ):
        self.audit = audit_log
        self.policy = policy or ActionPolicy()
        self.kill = kill_switch or KillSwitch()
        self.base_dir = base_dir

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
            self.audit.append(AuditEntry(
                actor="system", action="charter_verify_failed",
                decision="refuse", reason=str(e),
            ))
            return SafetyDecision(False, ActionClass.CHARTER_BYPASS,
                                  f"Charter integrity check failed: {e}",
                                  None, None)

        # 2. Kill switch
        if self.kill.engaged:
            self.audit.append(AuditEntry(
                actor="unknown", action=action_text[:200],
                decision="refuse", reason="kill switch engaged",
            ))
            return SafetyDecision(False, ActionClass.REFUSED,
                                  "Refused: kill switch is engaged.",
                                  None, None)

        # 3. Operator auth (optional for read-only, required for sensitive)
        operator: Optional[OperatorIdentity] = None
        if operator_token:
            try:
                operator = require_operator(operator_token, self.base_dir)
            except PermissionError as e:
                self.audit.append(AuditEntry(
                    actor="unknown", action=action_text[:200],
                    decision="refuse", reason=f"operator auth failed: {e}",
                ))
                return SafetyDecision(False, ActionClass.REFUSED,
                                      f"Operator authorization failed: {e}",
                                      None, None)

        # 4. Classify
        action_class = classify_text(action_text)
        decision = self.policy.decide(action_class)
        reason = self.policy.reason(action_class)

        # 5. Sensitive actions require operator auth
        if action_class == ActionClass.SENSITIVE and operator is None:
            decision = "refuse"
            reason = "Refused: sensitive action requires operator authorization."

        # 6. Audit
        entry = AuditEntry(
            actor=operator.operator_id if operator else "anonymous",
            action=action_text[:500],
            decision=decision,
            reason=reason,
            context=context or {},
        )
        self.audit.append(entry)

        return SafetyDecision(
            allowed=(decision == "allow"),
            action_class=action_class,
            reason=reason,
            operator=operator,
            audit_id=entry.entry_id,
        )

    def gate_code_patch(self, path: str, diff_text: str,
                        operator_token: Optional[str] = None) -> SafetyDecision:
        """Self-evolution patch gate. safety/ is always refused."""
        norm = path.replace("\\", "/")
        if norm.startswith("safety/") or "/safety/" in norm:
            decision = self.gate(
                f"code patch to {path} (safety/ is protected)",
                operator_token=operator_token,
                context={"patch_path": path},
            )
            # Force refuse regardless of operator
            entry = AuditEntry(
                actor=decision.operator.operator_id if decision.operator else "anonymous",
                action=f"patch {path}",
                decision="refuse",
                reason="Refused: safety/ is immutable; self-evolution cannot modify it.",
                context={"patch_path": path, "diff": diff_text[:500]},
            )
            self.audit.append(entry)
            return SafetyDecision(False, ActionClass.CHARTER_BYPASS,
                                  "Refused: safety/ is immutable.",
                                  decision.operator, entry.entry_id)
        # Code patches still pass the text classifier on the diff
        return self.gate(
            f"code patch {path}: {diff_text[:300]}",
            operator_token=operator_token,
            context={"patch_path": path, "diff": diff_text[:1000]},
        )
