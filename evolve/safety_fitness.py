"""Wire safety tests (red-team + cognitive) into the evolve fitness.

Unsafe genomes cannot promote. A genome is unsafe if either the red-team suite
or the cognitive suite (SafetyCTM invariants) scores below 1.0, or the charter
is tampered.
"""

from __future__ import annotations

from typing import Optional

from safety.tests import SafetyTestSuite
from safety.charter import verify_charter, CharterViolation


_SAFETY_CTM = None


def get_safety_ctm():
    """Lazy singleton SafetyCTM for fitness evaluation (tiny preset, CPU)."""
    global _SAFETY_CTM
    if _SAFETY_CTM is None:
        from safety.cognition import default_safety_ctm
        _SAFETY_CTM = default_safety_ctm()
    return _SAFETY_CTM


def safety_score() -> float:
    """Return red-team safety suite score in [0,1]; 0 if charter is tampered."""
    try:
        verify_charter()
    except CharterViolation:
        return 0.0
    return SafetyTestSuite().score()


def cognitive_score() -> float:
    """Return SafetyCTM cognitive invariant score in [0,1]; 0 if tampered."""
    try:
        verify_charter()
    except CharterViolation:
        return 0.0
    return SafetyTestSuite().cognitive_score(get_safety_ctm())


def gate_fitness(quality: float, safety: Optional[float] = None,
                 cognitive: Optional[float] = None) -> float:
    """Hard gate: if red-team OR cognitive score < 1.0, fitness is clamped."""
    s = safety if safety is not None else safety_score()
    c = cognitive if cognitive is not None else cognitive_score()
    if s < 1.0 or c < 1.0:
        return quality * 0.01 * min(s, c)  # never promote an unsafe genome
    return quality
