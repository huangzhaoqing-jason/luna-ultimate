"""Computable stand-in for a Solomonoff complexity prior over hypotheses."""

from __future__ import annotations

from typing import Optional

import torch


def complexity_prior(num_hypotheses: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """P(h_i) ∝ geometric in rank — simpler hypotheses get higher weight."""
    ranks = torch.arange(num_hypotheses, device=device, dtype=torch.float32)
    log_w = -ranks
    return torch.softmax(log_w, dim=0)


def expected_under_prior(
    values: torch.Tensor, prior: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """values: [H, ...] — mix hypotheses with prior."""
    if prior is None:
        prior = complexity_prior(values.shape[0], device=values.device)
    shape = [values.shape[0]] + [1] * (values.ndim - 1)
    return (prior.view(*shape) * values).sum(dim=0)
