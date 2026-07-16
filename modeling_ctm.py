"""Global Continuous Thought Module with JEPA for Luna Evolve."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import LunaConfig


class NeuronLevelModel(nn.Module):
    """Per-neuron processing MLP."""

    def __init__(self, n_neurons: int = 4096, hidden_dim: int = 128):
        super().__init__()
        self.n_neurons = n_neurons
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Parameter(torch.randn(n_neurons, 1, hidden_dim) * 0.02)
        self.fc1_bias = nn.Parameter(torch.zeros(n_neurons, hidden_dim))
        self.fc2 = nn.Parameter(torch.randn(n_neurons, hidden_dim, 1) * 0.02)
        self.fc2_bias = nn.Parameter(torch.zeros(n_neurons, 1))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Per-neuron MLP; x: [B, n_neurons] → [B, n_neurons]."""
        # fc1: [N, 1, H] — input dim is 1, so scale + bias
        w1 = self.fc1.squeeze(1)  # [N, H]
        h = x.unsqueeze(-1) * w1.unsqueeze(0) + self.fc1_bias.unsqueeze(0)
        h = F.silu(self.norm(h))
        w2 = self.fc2.squeeze(-1)  # [N, H]
        out = (h * w2.unsqueeze(0)).sum(dim=-1) + self.fc2_bias.squeeze(-1)
        return out


class SynchronyMatrix(nn.Module):
    """Cross-neuron correlation matrix."""

    def __init__(self, n_neurons: int = 4096):
        super().__init__()
        self.n_neurons = n_neurons
        self.temp = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, neuron_states: torch.Tensor) -> torch.Tensor:
        states_norm = F.normalize(neuron_states, dim=-1)
        sync = torch.bmm(states_norm.unsqueeze(-1), states_norm.unsqueeze(1))
        return sync * self.temp


class CTMJEPAPredictor(nn.Module):
    """Predict future neuron states from current state."""

    def __init__(self, n_neurons: int = 4096, hidden_dim: int = 256, predict_horizon: int = 1):
        super().__init__()
        self.n_neurons = n_neurons
        self.hidden_dim = hidden_dim
        self.predict_horizon = predict_horizon
        nhead = max(1, min(8, hidden_dim // 32))
        while hidden_dim % nhead != 0 and nhead > 1:
            nhead -= 1
        self.input_proj = nn.Linear(n_neurons, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dim_feedforward=hidden_dim * 4,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            for _ in range(2)
        ])
        self.output_proj = nn.Linear(hidden_dim, n_neurons)
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, current_state: torch.Tensor) -> torch.Tensor:
        h = self.input_norm(self.input_proj(current_state)).unsqueeze(1)
        for block in self.transformer_blocks:
            h = block(h)
        h = self.output_norm(h).squeeze(1)
        return self.output_proj(h)


class CTM(nn.Module):
    """Continuous Thought Module with future-state JEPA."""

    def __init__(self, config: LunaConfig):
        super().__init__()
        self.d_model = config.hidden_size
        self.n_neurons = config.ctm_n_neurons
        self.nlm_hidden = config.ctm_nlm_hidden
        self.max_ticks = config.ctm_max_ticks
        self.entropy_thresholds = config.ctm_entropy_thresholds
        self.use_jepa = getattr(config, "ctm_jepa_enabled", True)
        self.jepa_predict_horizon = getattr(config, "ctm_jepa_horizon", 1)
        self.jepa_ema_decay = getattr(config, "ctm_jepa_ema_decay", 0.996)

        self.synapse = nn.Linear(self.d_model, self.n_neurons, bias=False)
        self.nlm = NeuronLevelModel(self.n_neurons, self.nlm_hidden)
        self.sync_matrix = SynchronyMatrix(self.n_neurons)
        self.output_proj = nn.Linear(self.n_neurons, self.d_model, bias=False)
        self.gate = nn.Linear(self.d_model, 1, bias=False)
        self.norm = nn.LayerNorm(self.d_model)

        pred_hidden = min(256, max(32, self.n_neurons // 2))
        if self.use_jepa:
            self.jepa_predictor = CTMJEPAPredictor(
                n_neurons=self.n_neurons,
                hidden_dim=pred_hidden,
                predict_horizon=self.jepa_predict_horizon,
            )
            self.jepa_target = CTMJEPAPredictor(
                n_neurons=self.n_neurons,
                hidden_dim=pred_hidden,
                predict_horizon=self.jepa_predict_horizon,
            )
            self._init_target_predictor()
            self.register_buffer(
                "neuron_history", torch.zeros(1, self.max_ticks, self.n_neurons)
            )
        else:
            self.jepa_predictor = None
            self.jepa_target = None

    def _init_target_predictor(self):
        if self.jepa_target is None or self.jepa_predictor is None:
            return
        for target_param, pred_param in zip(
            self.jepa_target.parameters(), self.jepa_predictor.parameters()
        ):
            target_param.data.copy_(pred_param.data)
            target_param.requires_grad = False

    @torch.no_grad()
    def _update_target_predictor(self):
        if self.jepa_target is None or self.jepa_predictor is None:
            return
        for target_param, pred_param in zip(
            self.jepa_target.parameters(), self.jepa_predictor.parameters()
        ):
            target_param.data.mul_(self.jepa_ema_decay).add_(
                pred_param.data, alpha=1.0 - self.jepa_ema_decay
            )

    def compute_jepa_loss(
        self,
        current_state: torch.Tensor,
        future_state: torch.Tensor,
    ) -> torch.Tensor:
        """Predict future from current; target is EMA encoding of *future_state*."""
        if not self.use_jepa or self.jepa_predictor is None:
            return torch.zeros((), device=current_state.device)

        pred = self.jepa_predictor(current_state)
        with torch.no_grad():
            # Target encodes the actual future state (not current)
            target = self.jepa_target(future_state)

        pred_norm = F.normalize(pred, dim=-1)
        target_norm = F.normalize(target, dim=-1)
        cosine_loss = 1.0 - (pred_norm * target_norm).sum(dim=-1).mean()
        self._update_target_predictor()
        return cosine_loss

    def _compute_sync_entropy(self, sync: torch.Tensor) -> float:
        """Shannon entropy of |correlation| distribution (uncertainty signal)."""
        n = sync.shape[-1]
        mask = torch.triu(torch.ones(n, n, device=sync.device), diagonal=1).bool()
        flattened = sync[:, mask].abs().clamp_min(1e-8)
        # Softmax over corr magnitudes → distribution entropy
        probs = flattened / flattened.sum(dim=-1, keepdim=True)
        entropy = -(probs * probs.log()).sum(dim=-1).mean()
        # Normalize by log(n_pairs) to ~[0, 1]
        n_pairs = max(1, n * (n - 1) // 2)
        return float((entropy / math.log(n_pairs)).clamp(0, 1).item())

    def _determine_ticks(self, entropy: float) -> int:
        if entropy < self.entropy_thresholds[0]:
            return 1
        if entropy < self.entropy_thresholds[1]:
            return 2
        if entropy < self.entropy_thresholds[2]:
            return 3
        return self.max_ticks

    def forward(
        self,
        hidden_states: torch.Tensor,
        use_adaptive_early_exit: bool = True,
        return_jepa_loss: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, int, Optional[torch.Tensor]]:
        B, L, D = hidden_states.shape
        aggregated = hidden_states.mean(dim=1)
        neuron_state = self.synapse(aggregated)

        if use_adaptive_early_exit:
            init_sync = self.sync_matrix(neuron_state)
            entropy = self._compute_sync_entropy(init_sync)
            num_ticks = min(self.max_ticks, self._determine_ticks(entropy))
        else:
            num_ticks = self.max_ticks

        neuron_states = [neuron_state]
        sync_matrix = None

        for _ in range(num_ticks):
            neuron_state = neuron_state + self.nlm(neuron_state)
            sync_matrix = self.sync_matrix(neuron_state)
            # Cheap mixing: avoid full N×N matmul for large N — use sync mean
            if self.n_neurons <= 512:
                neuron_state = neuron_state + torch.bmm(
                    sync_matrix, neuron_state.unsqueeze(-1)
                ).squeeze(-1) * 0.1
            else:
                mix = sync_matrix.mean(dim=-1)
                neuron_state = neuron_state + mix * 0.1
            neuron_states.append(neuron_state)

        jepa_loss = None
        if return_jepa_loss and self.use_jepa and len(neuron_states) >= 2:
            jepa_loss = self.compute_jepa_loss(neuron_states[0], neuron_states[-1])

        ctm_global = self.output_proj(neuron_state)
        gate_value = torch.sigmoid(self.gate(aggregated))
        ctm_output = gate_value.unsqueeze(1) * ctm_global.unsqueeze(1)
        # Broadcast over sequence
        ctm_output = ctm_output.expand(B, L, D)

        if sync_matrix is None:
            sync_matrix = self.sync_matrix(neuron_state)

        return ctm_output, sync_matrix, num_ticks, jepa_loss

    def get_ctm_state(self, hidden_states: torch.Tensor) -> torch.Tensor:
        aggregated = hidden_states.mean(dim=1)
        return self.synapse(aggregated)
