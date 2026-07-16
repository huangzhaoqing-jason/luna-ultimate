"""Mamba2-SSD Layer for Luna Evolve (front layers)."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import LunaConfig
from quant_utils import RMSNorm


def _get_scan_fn():
    """Prefer RuntimeManager / luna_ops when available; keep autograd for train."""
    try:
        from luna_ops import _mamba2_scan_pytorch
        return _mamba2_scan_pytorch
    except Exception:
        return None


class Mamba2SSD(nn.Module):
    """Mamba2 Structured State Space Duality layer."""

    def __init__(self, config: LunaConfig):
        super().__init__()
        self.d_model = config.hidden_size
        self.d_state = config.mamba_d_state
        self.d_conv = config.mamba_d_conv
        self.expand = config.mamba_expand
        self.d_inner = self.d_model * self.expand
        self.d_inner_double = self.d_inner * 2

        self.norm = RMSNorm(self.d_model, eps=config.rms_norm_eps)
        self.in_proj = nn.Linear(self.d_model, self.d_inner_double, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv - 1,
        )
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).unsqueeze(0)
        A = A.repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.x_proj = nn.Linear(self.d_inner, self.d_inner + 2 * self.d_state, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def _selective_scan(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Selective scan returning (y, final_state). Preserves autograd."""
        scan_fn = _get_scan_fn()
        if scan_fn is not None and state is None:
            y, final_h = scan_fn(x, delta, A, B, C, D)
            return y, final_h

        # Manual path with optional warm-start state (needed for decode)
        B_size, L, d_inner = x.shape
        delta = delta.float()
        delta_A = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        delta_B = delta.unsqueeze(-1) * B.unsqueeze(2)
        if state is None:
            h = torch.zeros(
                B_size, d_inner, self.d_state, device=x.device, dtype=torch.float32
            )
        else:
            h = state.float()

        outputs = []
        for t in range(L):
            h = delta_A[:, t] * h + delta_B[:, t] * x[:, t].unsqueeze(-1).float()
            y_t = (h * C[:, t].unsqueeze(1).float()).sum(dim=-1)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)
        y = y + x.float() * D.unsqueeze(0).unsqueeze(0)
        return y.to(x.dtype), h.to(x.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        xz = self.in_proj(hidden_states)
        x, z = xz.chunk(2, dim=-1)

        x_conv = x.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[..., : x_conv.shape[-1]]
        x = F.silu(x_conv.transpose(1, 2))

        x_proj_out = self.x_proj(x)
        B_ssm = x_proj_out[:, :, : self.d_state]
        C_ssm = x_proj_out[:, :, self.d_state : 2 * self.d_state]
        dt_intermediate = x_proj_out[:, :, 2 * self.d_state :]
        delta = F.softplus(self.dt_proj(dt_intermediate))
        A = -torch.exp(self.A_log.float())

        # Prefer RuntimeManager dispatch at inference time
        if not torch.is_grad_enabled():
            try:
                from runtime_manager import get_runtime
                rt = get_runtime(verbose=False)
                y, final_h = rt.mamba2_scan(x, delta, A, B_ssm, C_ssm, self.D)
                if state is not None:
                    # Warm-start not supported in fused path; fall through
                    y, final_h = self._selective_scan(
                        x, delta, A, B_ssm, C_ssm, self.D, state=state
                    )
            except Exception:
                y, final_h = self._selective_scan(
                    x, delta, A, B_ssm, C_ssm, self.D, state=state
                )
        else:
            y, final_h = self._selective_scan(
                x, delta, A, B_ssm, C_ssm, self.D, state=state
            )

        y = y * F.silu(z)
        output = self.out_proj(y) + residual
        return output, final_h


class Mamba2Block(nn.Module):
    """Mamba2-SSD block with FlashMoE and CTM residual injection."""

    def __init__(self, config: LunaConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.mamba = Mamba2SSD(config)
        self.moe_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        moe_layer: nn.Module,
        ctm_residual: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        hidden_states, new_state = self.mamba(hidden_states, state)
        hidden_states = hidden_states + ctm_residual
        normed = self.moe_norm(hidden_states)
        hidden_states, aux_loss = moe_layer(normed, hidden_states)
        return hidden_states, aux_loss, new_state
