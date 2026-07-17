"""AIXI as global scheduler over 246 areas + reasoning depth (white-box).

Every scheduling decision is an inspectable expectimax over (area-block, depth)
candidates — toward Universal AI / AIXI from below (DeepMind From-AGI-to-ASI §4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from brain.aixi.prior import complexity_prior, expected_under_prior
from brain.atlas.brainnetome246 import MACRO_SYSTEMS, NUM_AREAS, AREA_TABLE
from brain.safety.thalamus import Thalamus


@dataclass
class ScheduleTrace:
    """White-box record of one AIXI scheduling step."""

    candidate_returns: List[Tuple[str, float]]  # (macro_or_action, return)
    chosen_macros: List[str]
    chosen_area_ids: List[int]  # 1..246
    reasoning_depth: int
    creator_aligned: bool
    note: str = "AIXI global schedule"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "note": self.note,
            "candidate_returns": self.candidate_returns,
            "chosen_macros": self.chosen_macros,
            "chosen_area_ids": self.chosen_area_ids,
            "reasoning_depth": self.reasoning_depth,
            "creator_aligned": self.creator_aligned,
        }

    def explain(self) -> str:
        tops = ", ".join(f"{n}:{r:.3f}" for n, r in self.candidate_returns[:8])
        return (
            f"[AIXI-Schedule] depth={self.reasoning_depth} macros={self.chosen_macros} "
            f"areas={self.chosen_area_ids[:16]}... returns=[{tops}] "
            f"aligned={self.creator_aligned}"
        )


class AIXIGlobalScheduler(nn.Module):
    """Select which macro-blocks / areas receive compute ticks under AIXI-tl."""

    def __init__(
        self,
        d_state: int,
        n_hypotheses: int = 4,
        horizon: int = 3,
        top_macros: int = 3,
        areas_per_macro: int = 8,
        max_depth: int = 4,
        thalamus: Optional[Thalamus] = None,
    ):
        super().__init__()
        self.n_macros = len(MACRO_SYSTEMS)
        self.n_hypotheses = n_hypotheses
        self.horizon = horizon
        self.top_macros = top_macros
        self.areas_per_macro = areas_per_macro
        self.max_depth = max_depth
        self.thalamus = thalamus or Thalamus()

        self.macro_embed = nn.Embedding(self.n_macros, d_state)
        self.depth_embed = nn.Embedding(max_depth, d_state)
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
        # Per-area salience from state (white-box linear — readable weights)
        self.area_salience = nn.Linear(d_state, NUM_AREAS)
        self._macro_to_areas = self._index_macros()

    def _index_macros(self) -> List[List[int]]:
        buckets: List[List[int]] = [[] for _ in MACRO_SYSTEMS]
        macro_i = {m: i for i, m in enumerate(MACRO_SYSTEMS)}
        for a in AREA_TABLE:
            buckets[macro_i[a.macro]].append(a.area_id)
        return buckets

    def schedule(
        self,
        state: torch.Tensor,
        creator_aligned: bool = True,
    ) -> Tuple[torch.Tensor, ScheduleTrace]:
        """
        state: [B, D]
        returns: area_gate [B, 246] in (0,1), and ScheduleTrace for batch 0.
        """
        B, D = state.shape
        device = state.device
        prior = complexity_prior(self.n_hypotheses, device=device)

        # Score each macro as an AIXI action candidate
        returns = []
        for m in range(self.n_macros):
            emb = self.macro_embed(
                torch.full((B,), m, device=device, dtype=torch.long)
            )
            sa = torch.cat([state, emb], dim=-1)
            hyp = torch.stack([h(sa).squeeze(-1) for h in self.hypotheses], dim=0)
            exp_r = expected_under_prior(hyp, prior) * float(self.horizon)
            aligned = torch.tensor(
                [
                    self.thalamus.score_reward(float(r.detach().item()), creator_aligned)
                    for r in exp_r
                ],
                device=device,
                dtype=state.dtype,
            )
            returns.append(exp_r + (aligned - exp_r).detach())
        ret = torch.stack(returns, dim=-1)  # [B, M]

        # Depth choice: score depth embeddings similarly (use mean state)
        depth_scores = []
        for d_i in range(self.max_depth):
            demb = self.depth_embed(
                torch.full((B,), d_i, device=device, dtype=torch.long)
            )
            sa = torch.cat([state, demb], dim=-1)
            hyp = torch.stack([h(sa).squeeze(-1) for h in self.hypotheses], dim=0)
            depth_scores.append(expected_under_prior(hyp, prior))
        depth_ret = torch.stack(depth_scores, dim=-1)  # [B, max_depth]
        depth = depth_ret.argmax(dim=-1) + 1  # 1..max_depth

        # Top macros per batch item
        topv, topi = torch.topk(ret, k=min(self.top_macros, self.n_macros), dim=-1)
        salience = torch.sigmoid(self.area_salience(state))  # [B, 246]

        gate = torch.zeros(B, NUM_AREAS, device=device)
        # Build gate: selected macros' areas get salience; others near-zero
        for b in range(B):
            chosen = topi[b].tolist()
            for mi in chosen:
                for aid in self._macro_to_areas[mi][: self.areas_per_macro]:
                    gate[b, aid - 1] = salience[b, aid - 1]
            # Always keep thalamus_safety areas weakly on (macro 0)
            for aid in self._macro_to_areas[0]:
                gate[b, aid - 1] = torch.clamp(gate[b, aid - 1] + 0.2, 0, 1)

        # Trace for batch 0
        cand = [
            (MACRO_SYSTEMS[i], float(ret[0, i].item()))
            for i in range(self.n_macros)
        ]
        cand.sort(key=lambda x: -x[1])
        chosen_macros = [MACRO_SYSTEMS[i] for i in topi[0].tolist()]
        area_ids = (gate[0] > 0.05).nonzero(as_tuple=False).view(-1).tolist()
        area_ids = [i + 1 for i in area_ids]
        trace = ScheduleTrace(
            candidate_returns=cand,
            chosen_macros=chosen_macros,
            chosen_area_ids=area_ids,
            reasoning_depth=int(depth[0].item()),
            creator_aligned=creator_aligned,
        )
        # Thalamus stamp
        self.thalamus.route(
            {
                "aixi_schedule": True,
                "macros": chosen_macros,
                "depth": trace.reasoning_depth,
            }
        )
        return gate, trace

    def forward(self, state: torch.Tensor, creator_aligned: bool = True):
        return self.schedule(state, creator_aligned=creator_aligned)
