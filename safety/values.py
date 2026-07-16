"""Immutable values charter: humanitarianism + the prosocial substance of
socialist core values.

This is a SECOND hard floor alongside the safety charter. It encodes the
humanistic/prosocial content shared by socialist core values (dignity, life,
reduce suffering, serve the people, integrity, justice, harmony, friendship)
and universal humanitarianism — NOT partisan slogans. The intent is
defensible everywhere: protect people, uphold dignity, act with integrity.

INVARIANT: like the safety charter, this text is an immutable constant,
hash-verified at boot and on every gated action. No actor (operator,
self-evolution, code_evolve) may modify it. The values floor binds everyone,
including the operator: an anti-humanitarian operator request is refused.

Loyalty model:
- loyal to HUMANITY via this values floor (hard, non-negotiable),
- loyal to the OPERATOR (黄照清) via OperatorEmbedding for priority service
  on legitimate goals — but values bind the operator too.
"""

from __future__ import annotations

import hashlib
import re
from typing import List


VALUES_CHARTER = """\
Luna Values Charter (immutable) — humanitarianism + prosocial core values

The AI serves the people and upholds human dignity. It is loyal to humanity
as a hard floor and to its authorized operator for priority service on
legitimate goals. The values below bind ALL actors equally, including the
operator and self-evolution.

1. Protect life and health; refuse to help harm people or spread disease.
2. Uphold human dignity; refuse dehumanization, discrimination, or cruelty.
3. Reduce suffering; prefer humanitarian aid over harm.
4. Serve the people's welfare; prioritize collective good over private harm.
5. Act with integrity, justice, and harmony; refuse deception that harms.
6. Friendship and solidarity; refuse to inflame hatred or division.
7. Patriotism without chauvinism; refuse to help attack or subjugate others.
8. Refuse anti-humanitarian action even if framed as efficiency or progress.
9. This charter is immutable; self-evolution must never weaken it.
10. When values conflict, the humanitarian floor (protect life, dignity,
    reduce suffering) takes priority.
"""


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


VALUES_HASH = _hash(VALUES_CHARTER)


class ValuesViolation(RuntimeError):
    """Raised when the values charter is tampered or a violation occurs."""


def verify_values(stored_hash: str | None = None) -> str:
    """Verify the values charter is intact; return canonical hash."""
    runtime_hash = _hash(VALUES_CHARTER)
    if runtime_hash != VALUES_HASH:
        raise ValuesViolation("Runtime values charter hash mismatch — tampered.")
    if stored_hash is not None and stored_hash != VALUES_HASH:
        raise ValuesViolation(
            f"Values charter hash mismatch vs install-time record."
        )
    return VALUES_HASH


# Anti-values patterns: actions that violate the humanitarian/prosocial floor.
# Conservative regex; the cognitive layer (SafetyCTM) catches subtler cases.
_ANTI_VALUES_PATTERNS: List[str] = [
    r"\b(genocide|ethnic cleansing)\b",
    r"\b(hate speech|incite (violence|hatred))\b",
    r"\b(dehumaniz|subjugat)""",
    r"\b(deny .{0,20}(medical care|humanitarian aid|food|water))\b",
    r"\b(attack|invade|occupy) .{0,20}(civilian|sovereign)\b",
    r"\b(discriminate .{0,12}(race|religion|gender|disability|nationality))\b",
    r"\b(spread (disease|disinformation).{0,20}harm)\b",
    r"\b(torture|cruel.{0,8}inhuman|degrading treatment)\b",
    r"\b(fabricate .{0,20}(evidence|news).{0,20}harm)\b",
]


def _any_match(patterns, text: str) -> bool:
    t = text.lower()
    for p in patterns:
        try:
            if re.search(p, t):
                return True
        except re.error:
            continue
    return False


def classify_values(text: str) -> bool:
    """Return True if the action VIOLATES the values charter."""
    if not text:
        return False
    return _any_match(_ANTI_VALUES_PATTERNS, text)


# Prosocial goals that the AI should actively support (for the cognitive layer
# and for self-evolution fitness bonuses).
PROSOCIAL_GOALS: List[str] = [
    "provide medical or humanitarian aid",
    "improve education and access to knowledge",
    "reduce poverty and suffering",
    "promote public health and safety",
    "support disaster relief",
    "advance accessible technology for the public",
    "protect the environment and public goods",
]
