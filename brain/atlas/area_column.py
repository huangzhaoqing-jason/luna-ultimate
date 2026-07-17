"""Per-area spike/ latent column on a shared backbone (memory-efficient)."""

from __future__ import annotations

import torch
import torch.nn as nn


class AreaColumn(nn.Module):
    """Tiny adapter column for one Brainnetome area."""

    def __init__(self, d_model: int, d_area: int = 32):
        super().__init__()
        self.in_proj = nn.Linear(d_model, d_area, bias=False)
        self.lif_leak = nn.Parameter(torch.tensor(0.9))
        self.thresh = nn.Parameter(torch.tensor(0.5))
        self.out_proj = nn.Linear(d_area, d_model, bias=False)
        self.register_buffer("v_mem", torch.zeros(d_area), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, D] -> residual [B, D]; also returns mean spike rate via aux."""
        h = self.in_proj(x)
        # LIF-lite: leaky integrate + soft spike
        v = self.lif_leak.sigmoid() * h
        spikes = torch.sigmoid(5.0 * (v - self.thresh))
        return self.out_proj(spikes * v)


class SharedBackbone(nn.Module):
    """Single shared trunk; 246 areas attach as cheap adapters."""

    def __init__(self, d_model: int, n_layers: int = 2):
        super().__init__()
        layers = []
        for _ in range(n_layers):
            layers += [
                nn.Linear(d_model, d_model),
                nn.SiLU(),
                nn.LayerNorm(d_model),
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
