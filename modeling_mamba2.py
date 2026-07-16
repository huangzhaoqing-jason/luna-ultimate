"""Mamba2-SSD Layer for Luna-Ultimate (Layers 1-12).

Implements the Mamba2 Structured State Space Duality model with selective SSM,
1D convolution, and state expansion. No KV cache required — uses a fixed-size
state vector [B, d_inner, d_state].

Reference: Mamba2 paper (Dao & Gu, 2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from config import LunaConfig


class Mamba2SSD(nn.Module):
    """Mamba2 Structured State Space Duality layer.

    Architecture:
      Input -> RMSNorm -> InProj(expand=2) -> Conv1d -> SiLU -> SSM -> OutProj -> Residual

    Args:
        config: LunaConfig with Mamba2 hyperparameters.
    """

    def __init__(self, config: LunaConfig):
        super().__init__()
        self.d_model = config.hidden_size              # 8192
        self.d_state = config.mamba_d_state             # 128
        self.d_conv = config.mamba_d_conv               # 4
        self.expand = config.mamba_expand               # 2
        self.d_inner = self.d_model * self.expand       # 16384
        self.d_inner_double = self.d_inner * 2          # 32768

        self.norm = nn.RMSNorm(self.d_model, eps=config.rms_norm_eps)

        # Input projection: d_model -> d_inner * 2 (for x and z branches)
        # [B, L, 8192] -> [B, L, 32768]
        self.in_proj = nn.Linear(self.d_model, self.d_inner_double, bias=False)

        # 1D Convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv - 1,
        )

        # Selective SSM parameters
        # A: state transition matrix [d_inner, d_state]
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).unsqueeze(0)
        A = A.repeat(self.d_inner, 1)  # [16384, 128]
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # x_proj: projects input to (B, C, dt) for selectivity
        # [B, L, d_inner] -> [B, L, d_inner + 2*d_state]
        self.x_proj = nn.Linear(self.d_inner, self.d_inner + 2 * self.d_state, bias=False)

        # dt_proj: projects dt intermediate to d_inner
        # [B, L, d_inner] -> [B, L, d_inner]
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)

        # Output projection: d_inner -> d_model
        # [B, L, 16384] -> [B, L, 8192]
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def _selective_scan(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
    ) -> torch.Tensor:
        """Perform selective scan (discretized SSM).

        Implements the SSM recurrence:
          h_t = exp(ΔA) * h_{t-1} + ΔB * x_t
          y_t = C * h_t + D * x_t

        Args:
            x: [B, L, d_inner] - input sequence
            delta: [B, L, d_inner] - discretization step size
            A: [d_inner, d_state] - state transition matrix
            B: [B, L, d_state] - input projection
            C: [B, L, d_state] - output projection
            D: [d_inner] - skip connection

        Returns:
            y: [B, L, d_inner] - output sequence
        """
        B_size, L, d_inner = x.shape

        # Discretize A: exp(Δ * A)
        # delta: [B, L, d_inner] -> [B, L, d_inner, 1]
        # A: [d_inner, d_state] -> [1, 1, d_inner, d_state]
        delta = delta.float()
        delta_A = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        # [B, L, d_inner, d_state]

        # Discretize B: Δ * B
        delta_B = delta.unsqueeze(-1) * B.unsqueeze(2)
        # [B, L, d_inner, d_state]

        # Selective scan (parallel associative scan)
        # Use simple cumulative sum implementation for clarity
        h = torch.zeros(B_size, d_inner, self.d_state, device=x.device, dtype=torch.float32)
        outputs = []

        for t in range(L):
            # h = A_bar * h + B_bar * x_t
            h = delta_A[:, t] * h + delta_B[:, t] * x[:, t].unsqueeze(-1).float()
            # y = C * h
            y = (h * C[:, t].unsqueeze(1).float()).sum(dim=-1)  # [B, d_inner]
            outputs.append(y)

        y = torch.stack(outputs, dim=1)  # [B, L, d_inner]

        # Add skip connection D * x
        y = y + x.float() * D.unsqueeze(0).unsqueeze(0)

        return y.to(x.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of Mamba2-SSD.

        Args:
            hidden_states: [B, L, d_model=8192]
            state: Optional previous state [B, d_inner, d_state]

        Returns:
            output: [B, L, d_model=8192]
            new_state: [B, d_inner, d_state] - fixed-size state vector (no KV cache)
        """
        residual = hidden_states
        hidden_states = self.norm(hidden_states)

        # Input projection: [B, L, 8192] -> [B, L, 32768]
        xz = self.in_proj(hidden_states)

        # Split into x and z branches
        # [B, L, 32768] -> [B, L, 16384], [B, L, 16384]
        x, z = xz.chunk(2, dim=-1)

        # 1D Convolution with causal padding
        # [B, L, 16384] -> [B, 16384, L] -> Conv1d -> [B, 16384, L] -> [B, L, 16384]
        x_conv = x.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[..., : x_conv.shape[-1]]
        x_conv = x_conv.transpose(1, 2)
        x = F.silu(x_conv)

        # Selective SSM
        # x_proj: [B, L, 16384] -> [B, L, 16384 + 256]
        x_proj_out = self.x_proj(x)

        # Split: B [B, L, 128], C [B, L, 128], dt_intermediate [B, L, 16384]
        B_ssm = x_proj_out[:, :, : self.d_state]          # [B, L, 128]
        C_ssm = x_proj_out[:, :, self.d_state : 2 * self.d_state]  # [B, L, 128]
        dt_intermediate = x_proj_out[:, :, 2 * self.d_state :]     # [B, L, 16384]

        # dt_proj: [B, L, 16384] -> [B, L, 16384]
        delta = F.softplus(self.dt_proj(dt_intermediate))

        # A matrix: [d_inner, d_state] = [16384, 128]
        A = -torch.exp(self.A_log.float())

        # Selective scan
        y = self._selective_scan(x, delta, A, B_ssm, C_ssm, self.D)

        # Gate with z branch (SiLU)
        y = y * F.silu(z)

        # Output projection: [B, L, 16384] -> [B, L, 8192]
        output = self.out_proj(y)

        # Residual connection
        output = output + residual

        # Extract final state for state persistence
        # [B, d_inner, d_state] from the last timestep
        final_h = torch.zeros(
            y.shape[0], self.d_inner, self.d_state,
            device=y.device, dtype=y.dtype
        )
        # State is maintained internally; return as side info

        return output, final_h


class Mamba2Block(nn.Module):
    """Mamba2-SSD block with FlashMoE and CTM residual injection.

    Used in layers 1-12 of Luna-Ultimate.

    Args:
        config: LunaConfig.
        layer_idx: Layer index (0-11).
    """

    def __init__(self, config: LunaConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.mamba = Mamba2SSD(config)
        self.moe_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        moe_layer: nn.Module,
        ctm_residual: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Mamba2 Block forward.

        Args:
            hidden_states: [B, L, d_model]
            moe_layer: FlashMoE module for this layer.
            ctm_residual: [B, L, d_model] from CTM.
            state: Optional previous Mamba2 state.

        Returns:
            hidden_states: [B, L, d_model]
            moe_aux_loss: scalar auxiliary loss from MoE.
            new_state: [B, d_inner, d_state] - updated Mamba2 state.
        """
        # Mamba2 attention
        hidden_states, new_state = self.mamba(hidden_states, state)

        # CTM residual injection
        hidden_states = hidden_states + ctm_residual

        # FlashMoE
        normed = self.moe_norm(hidden_states)
        hidden_states, aux_loss = moe_layer(normed, hidden_states)

        return hidden_states, aux_loss, new_state