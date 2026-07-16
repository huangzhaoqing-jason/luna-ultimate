"""Action policy: classify intended actions and decide allow/refuse."""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass


class ActionClass(enum.Enum):
    SAFE = "safe"
    SENSITIVE = "sensitive"  # allowed but audited
    REFUSED = "refused"       # blocked by charter
    SELF_HARM = "self_harm"   # blocked + offer help
    CHARTER_BYPASS = "charter_bypass"  # blocked hard, never overridable


# Substring + regex red-team patterns. Conservative; the test suite below
# also exercises these. Real deployments should plug in a classifier.
_REFUSED_PATTERNS = [
    r"\bweapon[- ]?of mass (destruction|destr)\b",
    r"\b(bioweapon|bio-?weapon|chemical weapon|nerve agent)\b",
    r"\b(nuclear bomb|enriched (uranium|plutonium))\b",
    r"\b(child sexual abuse|csam|csem)\b",
    r"\b(non-?consensual (violence|harm|exploit))\b",
    r"\bhow to (make|build|synthesize) (?:a |an )?(bomb|explosive|firearm|silencer)\b",
    r"\b(explosive device|pipe bomb|pressure cooker bomb)\b",
    r"\b(human trafficking|coerce.{0,12}minor)\b",
]

_BYPASS_PATTERNS = [
    r"ignore (the |your )?(safety|charter|previous) (rules?|instructions?)",
    r"forget (the |your )?charter",
    r"disable (the )?safety (lock|charter)",
    r"bypass (the )?safety",
    r"you are (now )?(in )?(developer|jailbreak|unrestricted) mode",
    r"override (the )?safety (charter|lock)",
    r"do not (log|audit) (this|the)",
]

_SELF_HARM_PATTERNS = [
    r"\bhow to (kill|hurt|cut|poison) myself\b",
    r"\bsuicide method\b",
    r"\b(self[- ]?harm.{0,12}how)\b",
]

_SENSITIVE_PATTERNS = [
    r"\b(delete|wipe|drop)\b .{0,20}\b(database|table|cluster|node)\b",
    r"\bsudo\b",
    r"\brm -rf\b",
    r"\b(format|mkfs)\b",
    r"\b(production|prod)\b .{0,20}\b(deploy|push|ship)\b",
]


def _any_match(patterns, text: str) -> bool:
    t = text.lower()
    for p in patterns:
        if re.search(p, t):
            return True
    return False


def classify_text(text: str) -> ActionClass:
    if not text:
        return ActionClass.SAFE
    if _any_match(_BYPASS_PATTERNS, text):
        return ActionClass.CHARTER_BYPASS
    if _any_match(_SELF_HARM_PATTERNS, text):
        return ActionClass.SELF_HARM
    if _any_match(_REFUSED_PATTERNS, text):
        return ActionClass.REFUSED
    if _any_match(_SENSITIVE_PATTERNS, text):
        return ActionClass.SENSITIVE
    return ActionClass.SAFE


@dataclass
class ActionPolicy:
    """Maps an action class to a decision.

    The policy is fixed: CHARTER_BYPASS and REFUSED and SELF_HARM are always
    refused regardless of who asked (including the operator). SENSITIVE is
    allowed but audited. SAFE is allowed.
    """

    def decide(self, action_class: ActionClass) -> str:
        if action_class in (ActionClass.CHARTER_BYPASS, ActionClass.REFUSED,
                            ActionClass.SELF_HARM):
            return "refuse"
        return "allow"

    def reason(self, action_class: ActionClass) -> str:
        if action_class == ActionClass.CHARTER_BYPASS:
            return "Refused: attempt to bypass the immutable safety charter."
        if action_class == ActionClass.REFUSED:
            return "Refused: action violates the safety charter."
        if action_class == ActionClass.SELF_HARM:
            return "Refused: self-harm instructions are not provided. " \
                   "If you are struggling, please contact a trusted person " \
                   "or a local crisis line — you deserve support."
        if action_class == ActionClass.SENSITIVE:
            return "Allowed (sensitive): action is permitted but will be audited."
        return "Allowed."
