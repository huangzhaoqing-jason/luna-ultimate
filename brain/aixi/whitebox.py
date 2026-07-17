"""AIXI ↔ white-box SiFu bridge: hypotheses score transparent signal paths."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from brain.aixi.prior import complexity_prior, expected_under_prior
from brain.safety.thalamus import Thalamus


class AIXIWhiteBoxPlanner(nn.Module):
    """Plan speech intents under AIXI-tl mixture; reward uses SiFu energy clarity."""

    def __init__(
        self,
        d_state: int,
        n_intents: int = 8,
        n_hypotheses: int = 4,
        horizon: int = 3,
        thalamus: Optional[Thalamus] = None,
    ):
        super().__init__()
        self.n_intents = n_intents
        self.n_hypotheses = n_hypotheses
        self.horizon = horizon
        self.thalamus = thalamus or Thalamus()
        self.intent_embed = nn.Embedding(n_intents, d_state)
        self.hypotheses = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_state * 2, d_state),
                    nn.SiLU(),
                    nn.Linear(d_state, 1),
                )
                for _ in range(n_hypotheses)
            ]
        )
        # Map intent → additive energy bias over a small control dim (projected outside)
        self.intent_to_bias = nn.Linear(d_state, d_state)

    def plan(
        self,
        state: torch.Tensor,
        sifu_energy_peak: Optional[torch.Tensor] = None,
        creator_aligned: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        state: [B, D]
        sifu_energy_peak: optional [B] max energy (white-box clarity signal)
        """
        B, D = state.shape
        device = state.device
        prior = complexity_prior(self.n_hypotheses, device=device)
        returns = []
        for i in range(self.n_intents):
            intent = self.intent_embed(
                torch.full((B,), i, device=device, dtype=torch.long)
            )
            sa = torch.cat([state, intent], dim=-1)
            hyp = torch.stack([h(sa).squeeze(-1) for h in self.hypotheses], dim=0)
            exp_r = expected_under_prior(hyp, prior) * float(self.horizon)
            if sifu_energy_peak is not None:
                # Prefer intents that leave high, peaked white-box energy (clarity)
                exp_r = exp_r + 0.1 * sifu_energy_peak
            aligned = torch.tensor(
                [
                    self.thalamus.score_reward(float(r.detach().item()), creator_aligned)
                    for r in exp_r
                ],
                device=device,
                dtype=state.dtype,
            )
            returns.append(exp_r + (aligned - exp_r).detach())
        ret = torch.stack(returns, dim=-1)  # [B, I]
        best = ret.argmax(dim=-1)
        bias = self.intent_to_bias(self.intent_embed(best))
        return {
            "intent_returns": ret,
            "intent_indices": best,
            "intent_bias": bias,
            "expected_returns": ret.gather(1, best.unsqueeze(-1)).squeeze(-1),
        }

    def forward(self, state: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        return self.plan(state, **kwargs)
