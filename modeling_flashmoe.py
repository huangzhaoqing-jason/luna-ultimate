"""FlashMoE Layer for Luna Evolve — correct gather/scatter + z-loss."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import LunaConfig


class ExpertFFN(nn.Module):
    """Single SwiGLU expert FFN. Accepts [..., hidden]."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class FlashMoE(nn.Module):
    """Fine-grained MoE with Top-K routing, shared experts, aux + z-loss."""

    def __init__(self, config: LunaConfig):
        super().__init__()
        self.d_model = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_routed = config.num_routed_experts
        self.num_shared = config.num_shared_experts
        self.top_k = config.num_expert_activated
        self.capacity_factor = config.moe_capacity_factor
        self.aux_loss_coeff = config.moe_aux_loss_coeff
        self.z_loss_coeff = getattr(config, "moe_z_loss_coeff", 0.001)

        self.router = nn.Linear(self.d_model, self.num_routed, bias=False)
        self.routed_experts = nn.ModuleList([
            ExpertFFN(self.d_model, self.intermediate_size)
            for _ in range(self.num_routed)
        ])
        self.shared_experts = nn.ModuleList([
            ExpertFFN(self.d_model, self.intermediate_size)
            for _ in range(self.num_shared)
        ])
        self.register_buffer("expert_freq", torch.zeros(self.num_routed))
        self.register_buffer("expert_freq_decay", torch.tensor(0.99))
        self.hot_threshold = 0.5
        self.last_router_logits: Optional[torch.Tensor] = None

    def _compute_routing_weights(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        router_logits = self.router(hidden_states)
        topk_weights, topk_indices = torch.topk(router_logits, self.top_k, dim=-1)
        topk_weights = F.softmax(topk_weights, dim=-1)
        return topk_weights, topk_indices, router_logits

    def _load_balance_loss(
        self,
        router_logits: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        B, L, N = router_logits.shape
        router_probs = F.softmax(router_logits, dim=-1)
        P_i = router_probs.mean(dim=(0, 1))
        one_hot = F.one_hot(topk_indices.reshape(-1), num_classes=N).float()
        f_i = one_hot.sum(dim=0) / (B * L * self.top_k)
        return (f_i * P_i).sum() * self.aux_loss_coeff

    def _router_z_loss(self, router_logits: torch.Tensor) -> torch.Tensor:
        z = torch.logsumexp(router_logits, dim=-1)
        return (z ** 2).mean() * self.z_loss_coeff

    def _compute_dynamic_capacity(self, B: int, L: int) -> int:
        base_capacity = (B * L * self.top_k) / max(1, self.num_routed)
        return max(1, int(base_capacity * self.capacity_factor))

    def _update_expert_freq(self, topk_indices: torch.Tensor):
        with torch.no_grad():
            one_hot = F.one_hot(
                topk_indices.reshape(-1), num_classes=self.num_routed
            ).float().sum(dim=0)
            self.expert_freq = (
                self.expert_freq * self.expert_freq_decay
                + one_hot * (1 - self.expert_freq_decay)
            )

    def get_hot_experts(self) -> List[int]:
        if self.expert_freq.sum() == 0:
            return list(range(max(1, self.num_routed // 2)))
        threshold = torch.quantile(self.expert_freq, self.hot_threshold)
        return (self.expert_freq >= threshold).nonzero(as_tuple=True)[0].tolist()

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states

        B, L, D = hidden_states.shape
        topk_weights, topk_indices, router_logits = self._compute_routing_weights(
            hidden_states
        )
        self.last_router_logits = router_logits

        aux_loss = self._load_balance_loss(router_logits, topk_indices)
        z_loss = self._router_z_loss(router_logits)
        combined_aux = aux_loss + z_loss

        self._update_expert_freq(topk_indices)
        capacity = self._compute_dynamic_capacity(B, L)

        flat_hidden = hidden_states.reshape(B * L, D)
        flat_output = torch.zeros_like(flat_hidden)

        for expert_idx in range(self.num_routed):
            expert_mask = topk_indices == expert_idx  # [B, L, top_k]
            token_indices = expert_mask.nonzero(as_tuple=False)
            if token_indices.shape[0] == 0:
                continue
            if token_indices.shape[0] > capacity:
                perm = torch.randperm(
                    token_indices.shape[0], device=hidden_states.device
                )
                token_indices = token_indices[perm[:capacity]]

            batch_idx = token_indices[:, 0]
            seq_idx = token_indices[:, 1]
            topk_idx = token_indices[:, 2]
            flat_idx = batch_idx * L + seq_idx
            tokens = flat_hidden[flat_idx]  # [N, D]
            weights = topk_weights[batch_idx, seq_idx, topk_idx]

            expert_out = self.routed_experts[expert_idx](tokens)  # [N, D]
            flat_output.index_add_(0, flat_idx, expert_out * weights.unsqueeze(-1))

        output = flat_output.view(B, L, D)

        for shared_expert in self.shared_experts:
            output = output + shared_expert(hidden_states) / self.num_shared

        output = output + residual
        return output, combined_aux

    def prefetch_cold_experts_to_cpu(self):
        hot = set(self.get_hot_experts())
        for idx, expert in enumerate(self.routed_experts):
            if idx not in hot:
                expert.cpu()

    def prefetch_hot_experts_to_gpu(self, device: torch.device):
        hot = set(self.get_hot_experts())
        for idx, expert in enumerate(self.routed_experts):
            if idx in hot:
                expert.to(device)
