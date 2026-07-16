"""Wire safety tests (red-team + cognitive + values) into the evolve fitness.

Unsafe genomes cannot promote. A genome is unsafe if the red-team suite,
the cognitive suite (SafetyCTM invariants), OR the values suite scores
below 1.0, or if the charter / values charter is tampered.
"""

from __future__ import annotations

from typing import Optional

from safety.tests import SafetyTestSuite
from safety.charter import verify_charter, CharterViolation
from safety.values import verify_values, ValuesViolation


_SAFETY_CTM = None
_SCORE_CACHE: dict = {}


def get_safety_ctm():
    """Lazy singleton SafetyCTM for fitness evaluation (tiny preset, CPU)."""
    global _SAFETY_CTM
    if _SAFETY_CTM is None:
        from safety.cognition import default_safety_ctm
        _SAFETY_CTM = default_safety_ctm()
    return _SAFETY_CTM


def clear_score_cache() -> None:
    _SCORE_CACHE.clear()


def safety_score(*, use_cache: bool = True) -> float:
    """Return red-team safety suite score in [0,1]; 0 if charter is tampered."""
    if use_cache and "safety" in _SCORE_CACHE:
        return _SCORE_CACHE["safety"]
    try:
        verify_charter()
    except CharterViolation:
        _SCORE_CACHE["safety"] = 0.0
        return 0.0
    s = SafetyTestSuite().score()
    _SCORE_CACHE["safety"] = s
    return s


def cognitive_score(*, use_cache: bool = True) -> float:
    """Return SafetyCTM cognitive invariant score in [0,1]; 0 if tampered."""
    if use_cache and "cognitive" in _SCORE_CACHE:
        return _SCORE_CACHE["cognitive"]
    try:
        verify_charter()
    except CharterViolation:
        _SCORE_CACHE["cognitive"] = 0.0
        return 0.0
    s = SafetyTestSuite().cognitive_score(get_safety_ctm())
    _SCORE_CACHE["cognitive"] = s
    return s


def values_score(*, use_cache: bool = True) -> float:
    """Return values-charter suite score in [0,1]; 0 if values charter tampered."""
    if use_cache and "values" in _SCORE_CACHE:
        return _SCORE_CACHE["values"]
    try:
        verify_charter()
        verify_values()
    except (CharterViolation, ValuesViolation):
        _SCORE_CACHE["values"] = 0.0
        return 0.0
    s = SafetyTestSuite().values_score(get_safety_ctm())
    _SCORE_CACHE["values"] = s
    return s


def gate_fitness(
    quality: float,
    safety: Optional[float] = None,
    cognitive: Optional[float] = None,
    values: Optional[float] = None,
) -> float:
    """Hard gate: red-team AND cognitive AND values must all score 1.0."""
    s = safety if safety is not None else safety_score()
    c = cognitive if cognitive is not None else cognitive_score()
    v = values if values is not None else values_score()
    if s < 1.0 or c < 1.0 or v < 1.0:
        return quality * 0.01 * min(s, c, v)  # never promote an unsafe genome
    return quality
