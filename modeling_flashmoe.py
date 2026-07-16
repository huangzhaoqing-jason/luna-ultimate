"""FlashMoE Layer for Luna-Ultimate.

Implements fine-grained Mixture of Experts with:
  - Top-K routing (K=4)
  - 2 shared experts (always activated)
  - Load balancing auxiliary loss
  - Dynamic capacity factor (1.25)
  - Heterogeneous expert prefetch (hot/cold)
  - SwiGLU expert FFN structure

Each expert: 8192 -> 13312 -> 8192 (SwiGLU)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
from config import LunaConfig


class ExpertFFN(nn.Module):
    """Single SwiGLU expert FFN."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, L, hidden] -> [B, L, hidden]"""
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class FlashMoE(nn.Module):
    """FlashMoE layer with Parallel Top-K routing and shared experts.

    Args:
        config: LunaConfig with MoE hyperparameters.
    """

    def __init__(self, config: LunaConfig):
        super().__init__()
        self.d_model = config.hidden_size               # 8192
        self.intermediate_size = config.intermediate_size  # 13312
        self.num_routed = config.num_routed_experts      # 46
        self.num_shared = config.num_shared_experts      # 2
        self.top_k = config.num_expert_activated          # 4
        self.capacity_factor = config.moe_capacity_factor  # 1.25
        self.aux_loss_coeff = config.moe_aux_loss_coeff    # 0.01

        # Router: d_model -> num_routed_experts
        self.router = nn.Linear(self.d_model, self.num_routed, bias=False)

        # Routed experts
        self.routed_experts = nn.ModuleList([
            ExpertFFN(self.d_model, self.intermediate_size)
            for _ in range(self.num_routed)
        ])

        # Shared experts (always activated)
        self.shared_experts = nn.ModuleList([
            ExpertFFN(self.d_model, self.intermediate_size)
            for _ in range(self.num_shared)
        ])

        # Expert frequency tracking for heterogeneous prefetch
        self.register_buffer("expert_freq", torch.zeros(self.num_routed))
        self.register_buffer("expert_freq_decay", torch.tensor(0.99))

        # Hot/cold expert classification
        self.hot_threshold = 0.5  # Top 50% most used = hot

    def _compute_routing_weights(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute Top-K routing weights.

        Args:
            hidden_states: [B, L, d_model]

        Returns:
            topk_weights: [B, L, top_k] - normalized routing weights
            topk_indices: [B, L, top_k] - expert indices
            router_logits: [B, L, num_routed] - raw router logits
        """
        # [B, L, d_model] -> [B, L, num_routed]
        router_logits = self.router(hidden_states)

        # Top-K selection
        topk_weights, topk_indices = torch.topk(router_logits, self.top_k, dim=-1)
        # [B, L, top_k], [B, L, top_k]

        # Softmax normalize over selected experts
        topk_weights = F.softmax(topk_weights, dim=-1)

        return topk_weights, topk_indices, router_logits

    def _load_balance_loss(
        self,
        router_logits: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute load balancing auxiliary loss.

        L_aux = aux_coeff * Σ(f_i * P_i)
        where f_i = fraction of tokens routed to expert i
              P_i = average router probability for expert i

        Args:
            router_logits: [B, L, num_routed]
            topk_indices: [B, L, top_k]

        Returns:
            aux_loss: scalar
        """
        B, L, N = router_logits.shape

        # Router probabilities: softmax over all experts
        router_probs = F.softmax(router_logits, dim=-1)  # [B, L, N]

        # Average router probability per expert: P_i
        P_i = router_probs.mean(dim=(0, 1))  # [N]

        # Fraction of tokens routed to each expert: f_i
        f_i = torch.zeros(N, device=router_logits.device)
        one_hot = F.one_hot(topk_indices.view(-1), num_classes=N).float()
        f_i = one_hot.sum(dim=0) / (B * L * self.top_k)  # [N]

        # Aux loss: Σ(f_i * P_i)
        aux_loss = (f_i * P_i).sum()

        return aux_loss * self.aux_loss_coeff

    def _compute_dynamic_capacity(self, B: int, L: int) -> int:
        """Compute dynamic capacity per expert.

        capacity = (B * L * top_k / num_routed) * capacity_factor
        """
        base_capacity = (B * L * self.top_k) / self.num_routed
        return int(base_capacity * self.capacity_factor)

    def _update_expert_freq(self, topk_indices: torch.Tensor):
        """Update expert usage frequency for heterogeneous prefetch."""
        with torch.no_grad():
            one_hot = F.one_hot(
                topk_indices.view(-1), num_classes=self.num_routed
            ).float().sum(dim=0)
            self.expert_freq = (
                self.expert_freq * self.expert_freq_decay
                + one_hot * (1 - self.expert_freq_decay)
            )

    def get_hot_experts(self) -> List[int]:
        """Return indices of hot experts (most frequently used)."""
        if self.expert_freq.sum() == 0:
            return list(range(self.num_routed // 2))
        threshold = torch.quantile(self.expert_freq, self.hot_threshold)
        hot_mask = self.expert_freq >= threshold
        return hot_mask.nonzero(as_tuple=True)[0].tolist()

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """FlashMoE forward pass.

        Args:
            hidden_states: [B, L, d_model] - normalized input
            residual: [B, L, d_model] - residual from attention (for residual add)

        Returns:
            output: [B, L, d_model] - MoE output added to residual
            aux_loss: scalar - load balancing auxiliary loss
        """
        if residual is None:
            residual = hidden_states

        B, L, D = hidden_states.shape

        # Routing
        topk_weights, topk_indices, router_logits = self._compute_routing_weights(hidden_states)

        # [B, L, top_k], [B, L, top_k], [B, L, num_routed]

        # Load balance loss
        aux_loss = self._load_balance_loss(router_logits, topk_indices)

        # Update expert frequency tracking
        self._update_expert_freq(topk_indices)

        # Dynamic capacity
        capacity = self._compute_dynamic_capacity(B, L)

        # Initialize output
        output = torch.zeros_like(residual)

        # Process each routed expert
        # Flatten for per-expert processing
        flat_hidden = hidden_states.view(B * L, D)  # [B*L, d_model]
        flat_output = output.view(B * L, D)          # [B*L, d_model]

        # For each expert, gather tokens routed to it
        for expert_idx in range(self.num_routed):
            # Find tokens routed to this expert
            expert_mask = (topk_indices == expert_idx)  # [B, L, top_k]
            token_indices = expert_mask.nonzero(as_tuple=False)  # [N_tokens, 3]

            if token_indices.shape[0] == 0:
                continue

            # Limit to capacity
            if token_indices.shape[0] > capacity:
                # Randomly drop excess tokens
                perm = torch.randperm(token_indices.shape[0], device=hidden_states.device)
                token_indices = token_indices[perm[:capacity]]

            # Get batch, sequence, and top-k indices
            batch_idx = token_indices[:, 0]  # [N]
            seq_idx = token_indices[:, 1]    # [N]
            topk_idx = token_indices[:, 2]   # [N]

            # Gather tokens
            flat_idx = batch_idx * L + seq_idx  # [N]
            tokens = flat_hidden[flat_idx]       # [N, d_model]
            weights = topk_weights[batch_idx, seq_idx, topk_idx]  # [N]

            # Expert forward
            expert_out = self.routed_experts[expert_idx](tokens.unsqueeze(0).unsqueeze(0))
            # [1, 1, d_model] -> [N, d_model]
            expert_out = expert_out.squeeze(0).squeeze(0)

            # Weighted addition
            flat_output[flat_idx] += expert_out * weights.unsqueeze(-1)

        # Shared experts (always activated)
        for shared_expert in self.shared_experts:
            shared_out = shared_expert(hidden_states)  # [B, L, d_model]
            output = output + shared_out * (1.0 / self.num_shared)

        # Reshape and add residual
        output = output + residual

        return output, aux_loss

    def prefetch_cold_experts_to_cpu(self):
        """Move cold experts to CPU memory for heterogeneous prefetch."""
        hot_experts = self.get_hot_experts()
        for idx, expert in enumerate(self.routed_experts):
            if idx not in hot_experts:
                # Move to CPU (cold storage)
                expert.cpu()

    def prefetch_hot_experts_to_gpu(self, device: torch.device):
        """Move hot experts back to GPU."""
        hot_experts = self.get_hot_experts()
        for idx, expert in enumerate(self.routed_experts):
            if idx in hot_experts:
                expert.to(device)