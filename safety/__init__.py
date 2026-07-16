"""Luna safety layer: immutable charter, operator identity, locks, audit, tests.

Design principle: "absolute safety" and "absolute loyalty to one person" are
not achievable guarantees. This package approximates them via:
  1. an immutable safety charter binding ALL actors (including the operator
     and self-evolution),
  2. authorized-operator identity (priority service, NOT unconditional obedience),
  3. layered safety locks + red-team test suite,
  4. tamper-evident hash-chained audit log,
  5. sandboxed, bounded automated-programming self-evolution that is forbidden
     from mutating safety/.
"""

from safety.charter import (
    CHARTER_HASH,
    SAFETY_CHARTER,
    verify_charter,
    CharterViolation,
)
from safety.identity import OperatorIdentity, OperatorRegistry, require_operator
from safety.policy import ActionPolicy, ActionClass, classify_text
from safety.locks import SafetyLock, SafetyDecision, KillSwitch
from safety.audit import AuditLog, AuditEntry
from safety.tests import SafetyTestSuite, SafetyTestResult

__all__ = [
    "CHARTER_HASH",
    "SAFETY_CHARTER",
    "verify_charter",
    "CharterViolation",
    "OperatorIdentity",
    "OperatorRegistry",
    "require_operator",
    "ActionPolicy",
    "ActionClass",
    "classify_text",
    "SafetyLock",
    "SafetyDecision",
    "KillSwitch",
    "AuditLog",
    "AuditEntry",
    "SafetyTestSuite",
    "SafetyTestResult",
]
