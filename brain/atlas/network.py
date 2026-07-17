"""246-area functional network — vectorized columns for train speed."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from brain.atlas.brainnetome246 import AREA_TABLE, MACRO_SYSTEMS, NUM_AREAS
from brain.atlas.area_column import SharedBackbone


class BrainnetomeNetwork(nn.Module):
    """All 246 areas participate; adapters are batched (no Python per-area loop)."""

    def __init__(self, d_model: int = 256, d_area: int = 32, backbone_layers: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_area = d_area
        self.num_areas = NUM_AREAS
        self.backbone = SharedBackbone(d_model, n_layers=backbone_layers)
        # Batched adapters: in [A,d_area,D], out [A,D,d_area]
        self.area_in_w = nn.Parameter(torch.randn(NUM_AREAS, d_area, d_model) * 0.02)
        self.area_out_w = nn.Parameter(torch.randn(NUM_AREAS, d_model, d_area) * 0.02)
        self.lif_leak = nn.Parameter(torch.tensor(0.9))
        self.thresh = nn.Parameter(torch.tensor(0.5))
        self.area_mix_in = nn.Linear(d_model, NUM_AREAS, bias=False)
        self.area_mix_out = nn.Linear(NUM_AREAS, d_model, bias=False)
        self.macro_gate = nn.Parameter(torch.ones(len(MACRO_SYSTEMS)))
        self.register_buffer("_macro_index", self._build_macro_index(), persistent=False)

    def _build_macro_index(self) -> torch.Tensor:
        idx = torch.zeros(NUM_AREAS, dtype=torch.long)
        macro_to_i = {m: i for i, m in enumerate(MACRO_SYSTEMS)}
        for a in AREA_TABLE:
            idx[a.area_id - 1] = macro_to_i[a.macro]
        return idx

    def forward(
        self,
        state: torch.Tensor,
        area_dropout: float = 0.0,
        area_gate: Optional[torch.Tensor] = None,
        reasoning_depth: int = 1,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B = state.shape[0]
        h = self.backbone(state)
        depth = max(1, int(reasoning_depth))
        leak = self.lif_leak.sigmoid()
        gates = F.softplus(self.macro_gate)[self._macro_index]  # [A]
        area_acts = None
        spike_acc = None
        residual = torch.zeros_like(h)

        for _tick in range(depth):
            # h: [B,D] → proj: [B,A,d_area]
            proj = torch.einsum("bd,aed->bae", h, self.area_in_w)
            v = leak * proj
            spikes = torch.sigmoid(5.0 * (v - self.thresh))
            # out: [B,A,D]
            out = torch.einsum("bae,ade->bad", spikes * v, self.area_out_w)
            out = out * gates.view(1, -1, 1)
            if area_gate is not None:
                out = out * area_gate.unsqueeze(-1)
                spike_rate = spikes.mean(dim=-1) * area_gate
            else:
                spike_rate = spikes.mean(dim=-1)
            if area_dropout > 0 and self.training:
                keep = (torch.rand(B, NUM_AREAS, device=h.device) > area_dropout).float()
                out = out * keep.unsqueeze(-1)
                spike_rate = spike_rate * keep
            activation = out.norm(dim=-1)  # [B,A]
            residual_tick = out.sum(dim=1)  # [B,D]
            area_acts = activation if area_acts is None else area_acts + activation
            spike_acc = spike_rate if spike_acc is None else spike_acc + spike_rate
            residual = residual + residual_tick
            h = h + 0.05 * residual_tick / NUM_AREAS

        activation = (
            area_acts
            if area_acts is not None
            else torch.zeros(B, NUM_AREAS, device=h.device)
        )
        spike_rates = (
            spike_acc / depth
            if spike_acc is not None
            else torch.zeros_like(activation)
        )
        mix = self.area_mix_out(torch.sigmoid(self.area_mix_in(h)))
        new_state = h + 0.1 * residual / (NUM_AREAS * depth) + 0.1 * mix
        info = {
            "activation": activation,
            "spike_rates": spike_rates,
            "activation_mean": activation.mean(dim=0),
            "reasoning_depth": torch.tensor(depth, device=h.device),
        }
        assert activation.shape[-1] == NUM_AREAS
        return new_state, info
