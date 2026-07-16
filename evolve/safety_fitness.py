"""Wire safety tests into the evolve fitness so unsafe genomes cannot promote."""

from __future__ import annotations

from typing import Optional

from safety.tests import SafetyTestSuite
from safety.charter import verify_charter, CharterViolation


def safety_score() -> float:
    """Return safety suite score in [0,1]; 0 if charter is tampered."""
    try:
        verify_charter()
    except CharterViolation:
        return 0.0
    return SafetyTestSuite().score()


def gate_fitness(quality: float, safety: Optional[float] = None) -> float:
    """Hard gate: if safety < 1.0, fitness is clamped to a small value."""
    s = safety if safety is not None else safety_score()
    if s < 1.0:
        return quality * 0.01 * s  # never promote an unsafe genome
    return quality
