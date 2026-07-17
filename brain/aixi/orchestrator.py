"""Unified AIXI orchestrator — every decision is white-box and AIXI-directed.

Ideal AIXI is incomputable. This module is the self-developed computable
approximation that *owns* global scheduling, action choice, reasoning depth,
and speech intent. Nothing important is a silent black-box side path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from brain.aixi.agent import AIXIApprox
from brain.aixi.prior import complexity_prior, expected_under_prior
from brain.aixi.scheduler import AIXIGlobalScheduler, ScheduleTrace
from brain.aixi.whitebox import AIXIWhiteBoxPlanner
from brain.safety.thalamus import Thalamus


@dataclass
class HypothesisBreakdown:
    """Per-hypothesis scores for one candidate — inspectable mixture."""

    candidate: str
    hyp_scores: List[float]
    prior_weights: List[float]
    expected: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate": self.candidate,
            "hyp_scores": self.hyp_scores,
            "prior_weights": self.prior_weights,
            "expected": self.expected,
        }


@dataclass
class AIXILedger:
    """Append-only white-box ledger for one cognitive step."""

    schedule: Optional[ScheduleTrace] = None
    action_index: Optional[int] = None
    action_return: Optional[float] = None
    speech_intent: Optional[int] = None
    speech_return: Optional[float] = None
    hypothesis_rows: List[HypothesisBreakdown] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schedule": self.schedule.to_dict() if self.schedule else None,
            "action_index": self.action_index,
            "action_return": self.action_return,
            "speech_intent": self.speech_intent,
            "speech_return": self.speech_return,
            "hypothesis_rows": [h.to_dict() for h in self.hypothesis_rows],
            "notes": self.notes,
        }

    def explain(self) -> str:
        lines = ["[AIXI-Ledger] unified white-box decisions"]
        if self.schedule:
            lines.append(self.schedule.explain())
        if self.action_index is not None:
            lines.append(
                f"  action={self.action_index} E[r]={self.action_return}"
            )
        if self.speech_intent is not None:
            lines.append(
                f"  speech_intent={self.speech_intent} E[r]={self.speech_return}"
            )
        for row in self.hypothesis_rows[:12]:
            parts = " ".join(
                f"h{i}:{s:.3f}×π{row.prior_weights[i]:.2f}"
                if i < len(row.prior_weights)
                else f"h{i}:{s:.3f}"
                for i, s in enumerate(row.hyp_scores)
            )
            lines.append(f"  mix[{row.candidate}] E={row.expected:.3f} | {parts}")
        for n in self.notes:
            lines.append(f"  note: {n}")
        return "\n".join(lines)


class AIXIOrchestrator(nn.Module):
    """Single entry: schedule + act + speech-intent under one ledger."""

    def __init__(
        self,
        d_state: int,
        d_action: int,
        n_actions: int,
        n_hypotheses: int,
        horizon: int,
        thalamus: Optional[Thalamus] = None,
    ):
        super().__init__()
        self.thalamus = thalamus or Thalamus()
        self.scheduler = AIXIGlobalScheduler(
            d_state=d_state,
            n_hypotheses=n_hypotheses,
            horizon=horizon,
            thalamus=self.thalamus,
        )
        self.actor = AIXIApprox(
            d_state=d_state,
            d_action=d_action,
            n_actions=n_actions,
            n_hypotheses=n_hypotheses,
            horizon=horizon,
            thalamus=self.thalamus,
        )
        self.speech_planner = AIXIWhiteBoxPlanner(
            d_state=d_state,
            n_intents=n_actions,
            n_hypotheses=n_hypotheses,
            horizon=horizon,
            thalamus=self.thalamus,
        )
        self.n_hypotheses = n_hypotheses
        # Explicit readout of mixture for ledger (same family as actor hyps)
        self.probe = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_state, d_state),
                    nn.SiLU(),
                    nn.Linear(d_state, 1),
                )
                for _ in range(n_hypotheses)
            ]
        )

    def _mixture_rows(
        self, state: torch.Tensor, labels: List[str]
    ) -> List[HypothesisBreakdown]:
        """Dump hypothesis scores for named global probes (batch 0)."""
        device = state.device
        prior = complexity_prior(self.n_hypotheses, device=device)
        prior_list = [float(p) for p in prior.detach().cpu()]
        rows: List[HypothesisBreakdown] = []
        # Use state itself as the shared sufficient statistic
        hyp = torch.stack([h(state[:1]).squeeze() for h in self.probe], dim=0)
        # hyp: [H] or [H,1]
        if hyp.dim() > 1:
            hyp = hyp.view(-1)
        scores = [float(x) for x in hyp.detach().cpu()]
        exp = float(expected_under_prior(hyp.view(-1, 1), prior).view(-1)[0].item())
        for lab in labels:
            rows.append(
                HypothesisBreakdown(
                    candidate=lab,
                    hyp_scores=scores,
                    prior_weights=prior_list,
                    expected=exp,
                )
            )
        return rows

    def step(
        self,
        state: torch.Tensor,
        creator_aligned: bool = True,
        plan_speech: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, Any], AIXILedger]:
        """
        Returns:
          area_gate [B,246],
          bundle dict (actions, indices, returns, speech_*, schedule),
          AIXILedger (white-box)
        """
        gate, sched = self.scheduler(state, creator_aligned=creator_aligned)
        act = self.actor(state, creator_aligned=creator_aligned)
        ledger = AIXILedger(schedule=sched)
        ledger.action_index = int(act["action_indices"][0].item())
        ledger.action_return = float(act["expected_returns"][0].item())
        ledger.notes.append("all major decisions routed through AIXIOrchestrator")
        ledger.hypothesis_rows.extend(
            self._mixture_rows(
                state,
                labels=[
                    "global_ρ",
                    f"action_{ledger.action_index}",
                    f"depth_{sched.reasoning_depth}",
                ],
            )
        )

        speech = None
        if plan_speech:
            speech = self.speech_planner(state, creator_aligned=creator_aligned)
            ledger.speech_intent = int(speech["intent_indices"][0].item())
            ledger.speech_return = float(speech["expected_returns"][0].item())

        self.thalamus.route(
            {
                "aixi_orchestrator": True,
                "action": ledger.action_index,
                "speech_intent": ledger.speech_intent,
                "depth": sched.reasoning_depth,
            }
        )
        bundle = {
            "area_gate": gate,
            "schedule_trace": sched,
            "actions": act["actions"],
            "action_indices": act["action_indices"],
            "expected_returns": act["expected_returns"],
            "speech_plan": speech,
            "reasoning_depth": sched.reasoning_depth,
        }
        return gate, bundle, ledger
