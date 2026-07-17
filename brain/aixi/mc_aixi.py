"""MC-AIXI style deeper planning with white-box rollout traces.

Finite horizon Monte-Carlo expectimax over discrete actions under a learned
hypothesis mixture — computable approximation toward Universal AI / AIXI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from brain.aixi.prior import complexity_prior, expected_under_prior
from brain.safety.thalamus import Thalamus


@dataclass
class RolloutStep:
    action: int
    hyp_rewards: List[float]
    expected_reward: float
    next_state_norm: float


@dataclass
class MCRolloutTrace:
    """Inspectable imagined trajectory under ρ."""

    horizon: int
    root_action: int
    total_expected: float
    steps: List[RolloutStep] = field(default_factory=list)
    prior_weights: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "horizon": self.horizon,
            "root_action": self.root_action,
            "total_expected": self.total_expected,
            "prior_weights": self.prior_weights,
            "steps": [
                {
                    "action": s.action,
                    "hyp_rewards": s.hyp_rewards,
                    "expected_reward": s.expected_reward,
                    "next_state_norm": s.next_state_norm,
                }
                for s in self.steps
            ],
        }

    def explain(self) -> str:
        lines = [
            f"[MC-AIXI] root_a={self.root_action} H={self.horizon} "
            f"E[G]={self.total_expected:.4f} π={self.prior_weights}"
        ]
        for i, s in enumerate(self.steps):
            lines.append(
                f"  t+{i+1} a={s.action} E[r]={s.expected_reward:.3f} "
                f"hyps={['%.2f'%x for x in s.hyp_rewards]} ||s||={s.next_state_norm:.3f}"
            )
        return "\n".join(lines)


class MCAIXIPlanner(nn.Module):
    """Deeper finite-horizon MC planning with hypothesis dynamics."""

    def __init__(
        self,
        d_state: int,
        n_actions: int = 8,
        n_hypotheses: int = 4,
        horizon: int = 5,
        n_samples: int = 4,
        thalamus: Optional[Thalamus] = None,
    ):
        super().__init__()
        self.d_state = d_state
        self.n_actions = n_actions
        self.n_hypotheses = n_hypotheses
        self.horizon = horizon
        self.n_samples = n_samples
        self.thalamus = thalamus or Thalamus()

        self.action_embed = nn.Embedding(n_actions, d_state)
        # Each hypothesis predicts (next_state_delta, reward)
        self.dyn = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_state * 2, d_state),
                    nn.SiLU(),
                    nn.Linear(d_state, d_state + 1),
                )
                for _ in range(n_hypotheses)
            ]
        )
        # Learnable log-prior (updated by experience via train step)
        self.log_prior = nn.Parameter(torch.zeros(n_hypotheses))

    def prior(self) -> torch.Tensor:
        return F.softmax(self.log_prior, dim=0)

    def update_posterior(self, hyp_errors: torch.Tensor, lr: float = 0.05) -> None:
        """hyp_errors: [H] — lower error → higher prior (Bayes-lite)."""
        with torch.no_grad():
            score = -hyp_errors
            self.log_prior.add_(lr * (score - score.mean()))

    def _transition(
        self, state: torch.Tensor, action: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        state: [B, D]
        returns next_states [H,B,D], rewards [H,B], mixed_next [B,D]
        """
        B = state.shape[0]
        a = self.action_embed(
            torch.full((B,), action, device=state.device, dtype=torch.long)
        )
        sa = torch.cat([state, a], dim=-1)
        outs = []
        for h in self.dyn:
            o = h(sa)
            outs.append(o)
        stacked = torch.stack(outs, dim=0)  # [H,B,D+1]
        deltas = stacked[..., :-1]
        rewards = stacked[..., -1]
        next_states = state.unsqueeze(0) + 0.1 * torch.tanh(deltas)
        prior = self.prior().to(state.device)
        mixed = (prior.view(-1, 1, 1) * next_states).sum(dim=0)
        return next_states, rewards, mixed

    def rollout(
        self,
        state: torch.Tensor,
        root_action: int,
        creator_aligned: bool = True,
    ) -> MCRolloutTrace:
        """One white-box MC trajectory starting with root_action (batch 0)."""
        prior = self.prior()
        prior_list = [float(x) for x in prior.detach().cpu()]
        s = state[:1]
        total = 0.0
        steps: List[RolloutStep] = []
        a = root_action
        discount = 1.0
        for _t in range(self.horizon):
            _ns, rewards, mixed = self._transition(s, a)
            # rewards [H,1]
            hyp_r = [float(rewards[h, 0].item()) for h in range(self.n_hypotheses)]
            exp_r = float((prior.to(s.device) * rewards[:, 0]).sum().item())
            if creator_aligned:
                exp_r = self.thalamus.score_reward(exp_r, True)
            total += discount * exp_r
            steps.append(
                RolloutStep(
                    action=a,
                    hyp_rewards=hyp_r,
                    expected_reward=exp_r,
                    next_state_norm=float(mixed[0].norm().item()),
                )
            )
            s = mixed
            # greedy continuation under mixture
            cont_scores = []
            for ca in range(self.n_actions):
                _, cr, _ = self._transition(s, ca)
                cont_scores.append(float((prior.to(s.device) * cr[:, 0]).sum().item()))
            a = int(max(range(self.n_actions), key=lambda i: cont_scores[i]))
            discount *= 0.95
        return MCRolloutTrace(
            horizon=self.horizon,
            root_action=root_action,
            total_expected=total,
            steps=steps,
            prior_weights=prior_list,
        )

    def plan(
        self, state: torch.Tensor, creator_aligned: bool = True
    ) -> Dict[str, Any]:
        """Evaluate all root actions via MC rollouts; pick max E[G]."""
        traces: List[MCRolloutTrace] = []
        scores = []
        for a in range(self.n_actions):
            # average a few stochastic samples via prior noise
            sample_vals = []
            last_tr = None
            for _ in range(self.n_samples):
                tr = self.rollout(state, a, creator_aligned=creator_aligned)
                last_tr = tr
                sample_vals.append(tr.total_expected)
            traces.append(last_tr)  # type: ignore[arg-type]
            scores.append(sum(sample_vals) / len(sample_vals))
        best = int(max(range(self.n_actions), key=lambda i: scores[i]))
        return {
            "action": best,
            "scores": scores,
            "traces": traces,
            "best_trace": traces[best],
            "prior": self.prior().detach(),
        }

    def forward(self, state: torch.Tensor, creator_aligned: bool = True) -> Dict[str, Any]:
        return self.plan(state, creator_aligned=creator_aligned)
