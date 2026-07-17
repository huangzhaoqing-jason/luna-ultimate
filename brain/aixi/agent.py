"""AIXI-tl / MC-AIXI style computable approximation (Universal AI from below).

Ideal AIXI is incomputable (Hutter / DeepMind From-AGI-to-ASI §4).
This agent maximizes expected reward under a finite hypothesis mixture and horizon.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from brain.aixi.prior import complexity_prior, expected_under_prior
from brain.safety.thalamus import Thalamus


@dataclass
class AIXIDecision:
    action_index: int
    expected_return: float
    action_vec: torch.Tensor


class AIXIApprox(nn.Module):
    """Finite-horizon expectimax over discrete action candidates."""

    def __init__(
        self,
        d_state: int,
        d_action: int,
        n_actions: int = 8,
        n_hypotheses: int = 4,
        horizon: int = 3,
        thalamus: Optional[Thalamus] = None,
    ):
        super().__init__()
        self.d_state = d_state
        self.d_action = d_action
        self.n_actions = n_actions
        self.n_hypotheses = n_hypotheses
        self.horizon = horizon
        self.thalamus = thalamus or Thalamus()

        # Hypothesis ensemble: each predicts next-state value from (s, a)
        self.hypotheses = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_state + d_action, d_state),
                    nn.SiLU(),
                    nn.Linear(d_state, 1),
                )
                for _ in range(n_hypotheses)
            ]
        )
        self.action_embed = nn.Embedding(n_actions, d_action)
        # Continuous action decoder from chosen discrete slot
        self.action_head = nn.Linear(d_state + d_action, d_action)

    def _predict_returns(
        self, state: torch.Tensor, creator_aligned: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return per-action expected returns [B, A] and best action embeds."""
        B = state.shape[0]
        device = state.device
        prior = complexity_prior(self.n_hypotheses, device=device)
        returns = []
        for a_idx in range(self.n_actions):
            a = self.action_embed(
                torch.full((B,), a_idx, device=device, dtype=torch.long)
            )
            sa = torch.cat([state, a], dim=-1)
            hyp_vals = torch.stack([h(sa).squeeze(-1) for h in self.hypotheses], dim=0)
            # [H, B] -> [B]
            exp_r = expected_under_prior(hyp_vals, prior)
            # Finite horizon discount proxy
            exp_r = exp_r * float(self.horizon)
            # Loyalty injection via thalamus
            aligned = torch.tensor(
                [
                    self.thalamus.score_reward(float(r.detach().item()), creator_aligned)
                    for r in exp_r
                ],
                device=device,
                dtype=state.dtype,
            )
            # Keep differentiable path: blend
            returns.append(exp_r + (aligned - exp_r).detach())
        return torch.stack(returns, dim=-1), self.action_embed.weight

    def decide(
        self, state: torch.Tensor, creator_aligned: bool = True
    ) -> List[AIXIDecision]:
        """Greedy argmax_a E[return | a] per batch element."""
        ret, _ = self._predict_returns(state, creator_aligned)
        best = ret.argmax(dim=-1)  # [B]
        out: List[AIXIDecision] = []
        for b in range(state.shape[0]):
            a_idx = int(best[b].item())
            a_emb = self.action_embed(
                torch.tensor([a_idx], device=state.device)
            )
            a_vec = self.action_head(torch.cat([state[b : b + 1], a_emb], dim=-1))
            out.append(
                AIXIDecision(
                    action_index=a_idx,
                    expected_return=float(ret[b, a_idx].item()),
                    action_vec=a_vec.squeeze(0),
                )
            )
        return out

    def forward(
        self, state: torch.Tensor, creator_aligned: bool = True
    ) -> dict:
        decisions = self.decide(state, creator_aligned=creator_aligned)
        actions = torch.stack([d.action_vec for d in decisions], dim=0)
        indices = torch.tensor(
            [d.action_index for d in decisions], device=state.device
        )
        returns = torch.tensor(
            [d.expected_return for d in decisions], device=state.device
        )
        return {
            "actions": actions,
            "action_indices": indices,
            "expected_returns": returns,
        }
