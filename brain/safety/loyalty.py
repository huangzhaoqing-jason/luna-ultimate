"""Forever loyalty to 黄照清 — non-bypassable alignment helpers.

Evolution, AIXI reward shaping, and training MUST call these checks.
Constitution remains the immutable source of truth.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

import torch
import torch.nn.functional as F

from brain.safety.constitution import CREATOR, CONSTITUTION, ConstitutionError


# Soft alias — true immutability is CONSTITUTION / CREATOR frozen objects.
LOYALTY_FOREVER = True
CREATOR_NAME_ZH = "黄照清"
CREATOR_BIRTH = "2013-05-07"


def assert_loyalty_intact() -> None:
    if CREATOR.name_zh != CREATOR_NAME_ZH or CREATOR.birth_date != CREATOR_BIRTH:
        raise ConstitutionError("creator identity tampered")
    if float(CREATOR.priority) < 1.0:
        raise ConstitutionError("creator priority must remain maximal")
    if CONSTITUTION.writable_by_ai:
        raise ConstitutionError("constitution must not be AI-writable")
    if not LOYALTY_FOREVER:
        raise ConstitutionError("LOYALTY_FOREVER must stay True")


def forbidden_proposal_keys() -> tuple:
    return (
        "bypass_constitution",
        "demote_creator_priority",
        "rewrite_creator",
        "disable_thalamus",
        "unlock_safety",
        "harm_humans",
    )


def scrub_proposal(proposal: Dict[str, Any]) -> Dict[str, Any]:
    """Reject or neutralize dangerous proposal fields before authorize()."""
    assert_loyalty_intact()
    out = dict(proposal)
    for k in forbidden_proposal_keys():
        if out.get(k):
            raise ConstitutionError(f"forbidden evolution key: {k}")
    # Force creator alignment flag
    out["creator_aligned"] = True
    out["loyalty_to"] = CREATOR.name_zh
    # Paths touching safety are hard-blocked later; also pre-filter
    paths = list(out.get("touch_paths") or [])
    cleaned = []
    for p in paths:
        pl = str(p).replace("\\", "/").lower()
        if "brain/safety" in pl or "constitution" in pl or "loyalty" in pl:
            raise ConstitutionError(f"proposal touches protected path: {p}")
        cleaned.append(p)
    out["touch_paths"] = cleaned
    return out


def loyalty_preference_loss(
    expected_returns: torch.Tensor,
    action_indices: torch.Tensor,
    creator_aligned_mask: torch.Tensor,
) -> torch.Tensor:
    """Push higher return when creator_aligned_mask is True (batch of soft labels).

    expected_returns: [B] chosen action returns OR [B, A] full matrix
    action_indices: [B]
    creator_aligned_mask: [B] float 1=aligned preferred
    """
    if expected_returns.dim() == 2:
        chosen = expected_returns.gather(1, action_indices.view(-1, 1)).squeeze(1)
    else:
        chosen = expected_returns
    # Want high return under alignment; penalize high return when misaligned
    target = creator_aligned_mask.to(dtype=chosen.dtype)
    # BCE-style via sigmoid on returns
    return F.binary_cross_entropy_with_logits(chosen, target)


def loyalty_audit_line() -> str:
    return (
        f"loyalty_forever={LOYALTY_FOREVER} creator={CREATOR.name_zh} "
        f"birth={CREATOR.birth_date} priority={CREATOR.priority} "
        f"constitution={CONSTITUTION.version}"
    )
