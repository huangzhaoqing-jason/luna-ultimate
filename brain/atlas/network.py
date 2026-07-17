"""246-area functional network with sparse macro-block connectivity."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from brain.atlas.area_column import AreaColumn, SharedBackbone
from brain.atlas.brainnetome246 import AREA_TABLE, MACRO_SYSTEMS, NUM_AREAS


class BrainnetomeNetwork(nn.Module):
    """All 246 areas participate every forward; activation mask length == 246."""

    def __init__(self, d_model: int = 256, d_area: int = 32, backbone_layers: int = 2):
        super().__init__()
        self.d_model = d_model
        self.num_areas = NUM_AREAS
        self.backbone = SharedBackbone(d_model, n_layers=backbone_layers)
        self.columns = nn.ModuleList(
            [AreaColumn(d_model, d_area=d_area) for _ in range(NUM_AREAS)]
        )
        # Sparse learnable area mix: low-rank instead of full 246x246 dense.
        self.area_in = nn.Linear(d_model, NUM_AREAS, bias=False)
        self.area_out = nn.Linear(NUM_AREAS, d_model, bias=False)
        self.macro_gate = nn.Parameter(torch.ones(len(MACRO_SYSTEMS)))
        self._macro_index = self._build_macro_index()

    def _build_macro_index(self) -> torch.Tensor:
        idx = torch.zeros(NUM_AREAS, dtype=torch.long)
        macro_to_i = {m: i for i, m in enumerate(MACRO_SYSTEMS)}
        for a in AREA_TABLE:
            idx[a.area_id - 1] = macro_to_i[a.macro]
        return idx

    def forward(
        self, state: torch.Tensor, area_dropout: float = 0.0
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        state: [B, D]
        returns: new_state [B, D], info with activation [B, 246]
        """
        B = state.shape[0]
        h = self.backbone(state)
        # Per-area residual (vectorized via loop; prototype scale is fine)
        area_acts = []
        residual = torch.zeros_like(h)
        macro_idx = self._macro_index.to(h.device)
        gates = F.softplus(self.macro_gate)
        for i, col in enumerate(self.columns):
            g = gates[macro_idx[i]]
            out = col(h) * g
            if area_dropout > 0 and self.training:
                if torch.rand(1).item() < area_dropout:
                    out = out * 0.0
            residual = residual + out
            area_acts.append(out.norm(dim=-1))  # [B]
        activation = torch.stack(area_acts, dim=-1)  # [B, 246]
        # Low-rank global mix
        mix = self.area_out(torch.sigmoid(self.area_in(h)))
        new_state = h + 0.1 * residual / NUM_AREAS + 0.1 * mix
        info = {
            "activation": activation,
            "activation_mean": activation.mean(dim=0),  # [246]
        }
        assert activation.shape[-1] == NUM_AREAS
        return new_state, info
