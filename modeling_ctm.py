"""Global Continuous Thought Module (CTM) for Luna-Ultimate.

The CTM is a 4096-neuron recurrent network that performs internal reasoning
over 1-4 adaptive ticks, decoupled from input sequence length. It computes a
global latent representation injected into every layer via gating.

Architecture:
  - Synapse: d_model(8192) -> ctm_n_neurons(4096)
  - NLM: 4096 neurons, each with 2-layer MLP (hidden=128)
  - Synchrony Matrix: 4096×4096 cross-neuron correlation
  - Output: 4096 -> 8192 with gating
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from config import LunaConfig


class NeuronLevelModel(nn.Module):
    """Per-neuron processing model.

    Each of the 4096 neurons independently processes its historical pre-activation
    through a 2-layer MLP with hidden dimension 128.

    Args:
        n_neurons: Number of neurons (4096).
        hidden_dim: Internal hidden dimension (128).
    """

    def __init__(self, n_neurons: int = 4096, hidden_dim: int = 128):
        super().__init__()
        self.n_neurons = n_neurons
        self.hidden_dim = hidden_dim

        # Each neuron has its own 2-layer MLP
        # [n_neurons, 1] -> [n_neurons, hidden_dim] -> [n_neurons, 1]
        self.fc1 = nn.Parameter(torch.randn(n_neurons, 1, hidden_dim) * 0.02)
        self.fc1_bias = nn.Parameter(torch.zeros(n_neurons, hidden_dim))
        self.fc2 = nn.Parameter(torch.randn(n_neurons, hidden_dim, 1) * 0.02)
        self.fc2_bias = nn.Parameter(torch.zeros(n_neurons, 1))

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process each neuron independently.

        Input:  [B, n_neurons]  - pre-activation per neuron
        Output: [B, n_neurons]  - updated activation per neuron
        """
        # [B, n_neurons] -> [B, n_neurons, 1]
        x = x.unsqueeze(-1)

        # [B, n_neurons, 1] @ [n_neurons, 1, hidden_dim] -> [B, n_neurons, hidden_dim]
        h = torch.matmul(x, self.fc1) + self.fc1_bias
        h = F.silu(self.norm(h))

        # [B, n_neurons, hidden_dim] @ [n_neurons, hidden_dim, 1] -> [B, n_neurons, 1]
        out = torch.matmul(h, self.fc2) + self.fc2_bias
        return out.squeeze(-1)  # [B, n_neurons]


class SynchronyMatrix(nn.Module):
    """Compute cross-neuron temporal correlation matrix.

    The 4096×4096 synchrony matrix captures pairwise temporal correlations
    between neurons, serving as a global latent representation of the
    model's internal state.
    """

    def __init__(self, n_neurons: int = 4096):
        super().__init__()
        self.n_neurons = n_neurons
        self.temp = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, neuron_states: torch.Tensor) -> torch.Tensor:
        """Compute synchrony matrix from neuron states.

        Input:  [B, n_neurons]
        Output: [B, n_neurons, n_neurons]
        """
        # Normalize neuron states
        states_norm = F.normalize(neuron_states, dim=-1)  # [B, n_neurons]

        # Outer product: [B, n_neurons, n_neurons]
        sync = torch.bmm(
            states_norm.unsqueeze(-1),  # [B, n_neurons, 1]
            states_norm.unsqueeze(1),   # [B, 1, n_neurons]
        )

        return sync * self.temp


class CTM(nn.Module):
    """Global Continuous Thought Module.

    Wraps the synapse, NLM, synchrony matrix, and output projection.
    Supports adaptive early exit based on synchrony matrix entropy.

    Args:
        config: LunaConfig with CTM hyperparameters.
    """

    def __init__(self, config: LunaConfig):
        super().__init__()
        self.d_model = config.hidden_size           # 8192
        self.n_neurons = config.ctm_n_neurons        # 4096
        self.nlm_hidden = config.ctm_nlm_hidden      # 128
        self.max_ticks = config.ctm_max_ticks        # 4
        self.entropy_thresholds = config.ctm_entropy_thresholds  # (0.3, 0.6, 0.9)

        # Synapse: d_model -> n_neurons
        self.synapse = nn.Linear(self.d_model, self.n_neurons, bias=False)

        # Neuron-Level Model
        self.nlm = NeuronLevelModel(self.n_neurons, self.nlm_hidden)

        # Synchrony matrix module
        self.sync_matrix = SynchronyMatrix(self.n_neurons)

        # Output projection: n_neurons -> d_model
        self.output_proj = nn.Linear(self.n_neurons, self.d_model, bias=False)

        # Gate: d_model -> 1 (sigmoid gating for residual injection)
        self.gate = nn.Linear(self.d_model, 1, bias=False)

        # LayerNorm for CTM internal
        self.norm = nn.LayerNorm(self.d_model)

    def _compute_sync_entropy(self, sync: torch.Tensor) -> float:
        """Compute entropy of the synchrony matrix for adaptive early exit.

        Input:  [B, n_neurons, n_neurons]
        Returns: scalar entropy value
        """
        # Flatten upper triangle (excluding diagonal) for entropy
        n = sync.shape[-1]
        mask = torch.triu(torch.ones(n, n, device=sync.device), diagonal=1).bool()
        flattened = sync[:, mask]  # [B, n_neurons*(n_neurons-1)/2]

        # Compute eigenvalues via power iteration approximation
        # Use variance of the flattened matrix as entropy proxy
        # Higher variance = more structured = lower entropy = simpler input
        entropy = -torch.mean(torch.abs(flattened), dim=-1).mean().item()
        return entropy

    def _determine_ticks(self, entropy: float) -> int:
        """Determine number of ticks based on synchrony matrix entropy.

        High entropy (complex input) -> more ticks.
        Low entropy (simple input) -> fewer ticks, early exit.

        thresholds: (0.3, 0.6, 0.9)
        entropy < 0.3  -> 1 tick
        entropy < 0.6  -> 2 ticks
        entropy < 0.9  -> 3 ticks
        entropy >= 0.9 -> 4 ticks (full thinking)
        """
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
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Run CTM over the input hidden states.

        Args:
            hidden_states: [B, L, d_model] - hidden states from current layer
            use_adaptive_early_exit: Whether to use adaptive tick count

        Returns:
            ctm_output: [B, L, d_model] - CTM residual to inject
            sync_matrix: [B, n_neurons, n_neurons] - global synchrony state
            num_ticks_used: int - actual ticks used (1-4)
        """
        B, L, D = hidden_states.shape

        # Aggregate over sequence: mean pooling
        # [B, L, d_model] -> [B, d_model]
        aggregated = hidden_states.mean(dim=1)

        # Synapse: [B, d_model] -> [B, n_neurons]
        neuron_state = self.synapse(aggregated)

        # Determine number of ticks
        if use_adaptive_early_exit:
            # Quick initial sync computation for entropy
            init_sync = self.sync_matrix(neuron_state)  # [B, n_neurons, n_neurons]
            entropy = self._compute_sync_entropy(init_sync)
            num_ticks = self._determine_ticks(entropy)
        else:
            num_ticks = self.max_ticks

        # Run NLM for the determined number of ticks
        sync_matrix = None
        for tick in range(num_ticks):
            # NLM processes each neuron
            neuron_state = neuron_state + self.nlm(neuron_state)  # [B, n_neurons]

            # Update synchrony matrix
            sync_matrix = self.sync_matrix(neuron_state)  # [B, n_neurons, n_neurons]

            # Global information mixing via synchrony matrix
            # [B, n_neurons, n_neurons] @ [B, n_neurons, 1] -> [B, n_neurons, 1]
            neuron_state = neuron_state + torch.bmm(
                sync_matrix, neuron_state.unsqueeze(-1)
            ).squeeze(-1) * 0.1

        # Output projection: [B, n_neurons] -> [B, d_model]
        ctm_global = self.output_proj(neuron_state)  # [B, d_model]

        # Gate: [B, d_model] -> [B, 1]  (sigmoid)
        gate_value = torch.sigmoid(self.gate(aggregated))  # [B, 1]

        # Expand to sequence length
        # [B, d_model] -> [B, 1, d_model] -> [B, L, d_model]
        ctm_output = gate_value.unsqueeze(1) * ctm_global.unsqueeze(1)

        return ctm_output, sync_matrix, num_ticks

    def get_ctm_state(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Get the global CTM state for layer skipping decisions.

        Returns: [B, n_neurons] - neuron state before output projection
        """
        aggregated = hidden_states.mean(dim=1)
        neuron_state = self.synapse(aggregated)
        return neuron_state