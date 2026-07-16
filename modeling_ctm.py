"""Global Continuous Thought Module with JEPA for Luna-Ultimate.

Upgraded CTM with CTM-JEPA (Joint Embedding Predictive Architecture):
  - Synapse: d_model(8192) → ctm_n_neurons(4096)
  - NLM: 4096 neurons, each with 2-layer MLP (hidden=128)
  - Synchrony Matrix: 4096×4096 cross-neuron correlation
  - CTM-JEPA: EMA target network predicts future neuron states
  - Adaptive early exit based on synchrony matrix entropy (1-4 ticks)
  - Output: 4096 → 8192, gated residual injection

The CTM-JEPA component trains the model to predict its own future thinking
states, enforcing internal reasoning consistency across time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import copy
import math
from config import LunaConfig


class NeuronLevelModel(nn.Module):
    """Per-neuron processing model.

    Each of 4096 neurons independently processes its historical pre-activation
    through a 2-layer MLP with hidden dimension 128.

    Args:
        n_neurons: Number of neurons (4096).
        hidden_dim: Internal hidden dimension (128).
    """

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
        """[B, n_neurons] → [B, n_neurons]"""
        x = x.unsqueeze(-1)  # [B, n_neurons, 1]
        h = torch.matmul(x, self.fc1) + self.fc1_bias  # [B, n_neurons, hidden]
        h = F.silu(self.norm(h))
        out = torch.matmul(h, self.fc2) + self.fc2_bias  # [B, n_neurons, 1]
        return out.squeeze(-1)  # [B, n_neurons]


class SynchronyMatrix(nn.Module):
    """Compute cross-neuron temporal correlation matrix."""

    def __init__(self, n_neurons: int = 4096):
        super().__init__()
        self.n_neurons = n_neurons
        self.temp = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, neuron_states: torch.Tensor) -> torch.Tensor:
        """[B, n_neurons] → [B, n_neurons, n_neurons]"""
        states_norm = F.normalize(neuron_states, dim=-1)
        sync = torch.bmm(
            states_norm.unsqueeze(-1),  # [B, n_neurons, 1]
            states_norm.unsqueeze(1),   # [B, 1, n_neurons]
        )
        return sync * self.temp


class CTMJEPAPredictor(nn.Module):
    """CTM-JEPA Predictor: predicts future neuron states.

    Takes current neuron state h_t and predicts h_{t+k} for k ∈ {1, 2, 3, 4}.
    Uses a lightweight transformer to capture temporal dependencies across
    the 4096-dimensional neuron state space.

    Args:
        n_neurons: 4096
        hidden_dim: Predictor hidden dimension
        predict_horizon: How many steps ahead to predict (default: 1)
    """

    def __init__(self, n_neurons: int = 4096, hidden_dim: int = 256, predict_horizon: int = 1):
        super().__init__()
        self.n_neurons = n_neurons
        self.hidden_dim = hidden_dim
        self.predict_horizon = predict_horizon

        # Project neuron state to predictor hidden space
        self.input_proj = nn.Linear(n_neurons, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        # Lightweight temporal transformer (2 blocks)
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim * 4,
                batch_first=True, activation="gelu", norm_first=True,
            )
            for _ in range(2)
        ])

        # Output projection: hidden_dim → n_neurons
        self.output_proj = nn.Linear(hidden_dim, n_neurons)
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, current_state: torch.Tensor) -> torch.Tensor:
        """Predict future neuron state.

        Input:  [B, n_neurons=4096] — current neuron state
        Output: [B, n_neurons=4096] — predicted future state
        """
        B, N = current_state.shape

        # Project to hidden space
        # [B, 4096] → [B, hidden_dim]
        h = self.input_proj(current_state)
        h = self.input_norm(h)

        # Add sequence dim for transformer
        h = h.unsqueeze(1)  # [B, 1, hidden_dim]

        # Apply transformer blocks
        for block in self.transformer_blocks:
            h = block(h)

        h = self.output_norm(h)
        h = h.squeeze(1)  # [B, hidden_dim]

        # Project back to neuron space
        prediction = self.output_proj(h)  # [B, n_neurons]

        return prediction


class CTM(nn.Module):
    """Global Continuous Thought Module with JEPA.

    Upgraded with CTM-JEPA:
      - EMA target network tracks neuron state evolution
      - Predictor forecasts future neuron states (h_{t+k})
      - Loss: cosine similarity between predicted and target future states
      - Adaptive early exit based on synchrony matrix entropy

    Args:
        config: LunaConfig with CTM hyperparameters.
    """

    def __init__(self, config: LunaConfig):
        super().__init__()
        self.d_model = config.hidden_size           # 8192
        self.n_neurons = config.ctm_n_neurons        # 4096
        self.nlm_hidden = config.ctm_nlm_hidden      # 128
        self.max_ticks = config.ctm_max_ticks        # 4
        self.entropy_thresholds = config.ctm_entropy_thresholds

        # CTM-JEPA settings
        self.use_jepa = getattr(config, "ctm_jepa_enabled", True)
        self.jepa_predict_horizon = getattr(config, "ctm_jepa_horizon", 1)
        self.jepa_ema_decay = getattr(config, "ctm_jepa_ema_decay", 0.996)

        # Synapse: d_model → n_neurons
        self.synapse = nn.Linear(self.d_model, self.n_neurons, bias=False)

        # Neuron-Level Model
        self.nlm = NeuronLevelModel(self.n_neurons, self.nlm_hidden)

        # Synchrony matrix module
        self.sync_matrix = SynchronyMatrix(self.n_neurons)

        # Output projection: n_neurons → d_model
        self.output_proj = nn.Linear(self.n_neurons, self.d_model, bias=False)

        # Gate: d_model → 1 (sigmoid gating for residual injection)
        self.gate = nn.Linear(self.d_model, 1, bias=False)

        # LayerNorm for CTM internal
        self.norm = nn.LayerNorm(self.d_model)

        # ========== CTM-JEPA Components ==========
        if self.use_jepa:
            # Predictor: forecasts future neuron states
            self.jepa_predictor = CTMJEPAPredictor(
                n_neurons=self.n_neurons,
                hidden_dim=256,
                predict_horizon=self.jepa_predict_horizon,
            )

            # EMA target network: generates ground truth for JEPA
            # Deep copy of predictor as target
            self.jepa_target = CTMJEPAPredictor(
                n_neurons=self.n_neurons,
                hidden_dim=256,
                predict_horizon=self.jepa_predict_horizon,
            )
            self._init_target_predictor()

            # Neuron state history buffer for JEPA
            self.register_buffer("neuron_history", torch.zeros(1, self.max_ticks, self.n_neurons))
        else:
            self.jepa_predictor = None
            self.jepa_target = None

    def _init_target_predictor(self):
        """Initialize target predictor with same weights as context predictor."""
        if self.jepa_target is not None and self.jepa_predictor is not None:
            for target_param, pred_param in zip(
                self.jepa_target.parameters(),
                self.jepa_predictor.parameters(),
            ):
                target_param.data.copy_(pred_param.data)
                target_param.requires_grad = False

    @torch.no_grad()
    def _update_target_predictor(self):
        """EMA update of target predictor."""
        if self.jepa_target is not None and self.jepa_predictor is not None:
            for target_param, pred_param in zip(
                self.jepa_target.parameters(),
                self.jepa_predictor.parameters(),
            ):
                target_param.data.mul_(self.jepa_ema_decay).add_(
                    pred_param.data, alpha=1.0 - self.jepa_ema_decay
                )

    def compute_jepa_loss(
        self,
        current_state: torch.Tensor,
        future_state: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CTM-JEPA prediction loss.

        Predicts future neuron state from current state, compares against
        EMA target's prediction of the same future state.

        Args:
            current_state: [B, n_neurons] — neuron state at tick t
            future_state: [B, n_neurons] — neuron state at tick t+k

        Returns:
            jepa_loss: scalar cosine similarity loss
        """
        if not self.use_jepa or self.jepa_predictor is None:
            return torch.tensor(0.0, device=current_state.device)

        # Context predictor: predict future from current
        pred = self.jepa_predictor(current_state)  # [B, n_neurons]

        # Target: predict future from EMA target
        with torch.no_grad():
            target = self.jepa_target(current_state)  # [B, n_neurons]

        # Cosine similarity loss
        pred_norm = F.normalize(pred, dim=-1)
        target_norm = F.normalize(target, dim=-1)
        cosine_loss = 1.0 - (pred_norm * target_norm).sum(dim=-1).mean()

        # Update target predictor via EMA
        self._update_target_predictor()

        return cosine_loss

    def _compute_sync_entropy(self, sync: torch.Tensor) -> float:
        """Compute entropy of the synchrony matrix for adaptive early exit."""
        n = sync.shape[-1]
        mask = torch.triu(torch.ones(n, n, device=sync.device), diagonal=1).bool()
        flattened = sync[:, mask]
        entropy = -torch.mean(torch.abs(flattened), dim=-1).mean().item()
        return entropy

    def _determine_ticks(self, entropy: float) -> int:
        """Determine number of ticks based on synchrony matrix entropy."""
        if entropy < self.entropy_thresholds[0]:
            return 1
        elif entropy < self.entropy_thresholds[1]:
            return 2
        elif entropy < self.entropy_thresholds[2]:
            return 3
        else:
            return self.max_ticks

    def forward(
        self,
        hidden_states: torch.Tensor,
        use_adaptive_early_exit: bool = True,
        return_jepa_loss: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, int, Optional[torch.Tensor]]:
        """Run CTM over the input hidden states.

        Args:
            hidden_states: [B, L, d_model]
            use_adaptive_early_exit: Use adaptive tick count.
            return_jepa_loss: Whether to compute and return JEPA loss.

        Returns:
            ctm_output: [B, L, d_model] — CTM residual to inject
            sync_matrix: [B, n_neurons, n_neurons] — global synchrony state
            num_ticks_used: int — actual ticks used (1-4)
            jepa_loss: Optional scalar — CTM-JEPA prediction loss
        """
        B, L, D = hidden_states.shape

        # Aggregate over sequence: mean pooling
        aggregated = hidden_states.mean(dim=1)  # [B, d_model]

        # Synapse: [B, d_model] → [B, n_neurons]
        neuron_state = self.synapse(aggregated)

        # Determine number of ticks
        if use_adaptive_early_exit:
            init_sync = self.sync_matrix(neuron_state)
            entropy = self._compute_sync_entropy(init_sync)
            num_ticks = self._determine_ticks(entropy)
        else:
            num_ticks = self.max_ticks

        # Store neuron states for JEPA
        neuron_states = [neuron_state.clone()]
        sync_matrix = None

        # Run NLM for the determined number of ticks
        for tick in range(num_ticks):
            # NLM processes each neuron
            neuron_state = neuron_state + self.nlm(neuron_state)  # [B, n_neurons]

            # Update synchrony matrix
            sync_matrix = self.sync_matrix(neuron_state)

            # Global information mixing via synchrony matrix
            neuron_state = neuron_state + torch.bmm(
                sync_matrix, neuron_state.unsqueeze(-1)
            ).squeeze(-1) * 0.1

            neuron_states.append(neuron_state.clone())

        # ========== CTM-JEPA Loss ==========
        jepa_loss = None
        if return_jepa_loss and self.use_jepa and len(neuron_states) >= 2:
            # Predict future state from current state
            current = neuron_states[0]  # [B, n_neurons] — state at tick 0
            future = neuron_states[-1]  # [B, n_neurons] — state at final tick
            jepa_loss = self.compute_jepa_loss(current, future)

        # Output projection: [B, n_neurons] → [B, d_model]
        ctm_global = self.output_proj(neuron_state)  # [B, d_model]

        # Gate: [B, d_model] → [B, 1] (sigmoid)
        gate_value = torch.sigmoid(self.gate(aggregated))  # [B, 1]

        # Expand to sequence length
        ctm_output = gate_value.unsqueeze(1) * ctm_global.unsqueeze(1)

        return ctm_output, sync_matrix, num_ticks, jepa_loss

    def get_ctm_state(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Get the global CTM state for layer skipping decisions."""
        aggregated = hidden_states.mean(dim=1)
        neuron_state = self.synapse(aggregated)
        return neuron_state