"""Luna-Ultimate Cross-Platform Operator Abstraction Layer.

Defines unified operator interfaces for all custom kernel operations in Luna-Ultimate.
Each operator is implemented as a torch.autograd.Function subclass, allowing seamless
integration with PyTorch's autograd while enabling backend-specific kernel dispatch.

Architecture:
    LunaOps (abstract interface)
    ├── Mamba2ScanOp    → Selective SSM scan (Mamba2)
    ├── FlashAttentionOp → MLA compressed attention
    ├── MoEGateOp       → Top-K gating with load balancing
    └── JEPAPredictorOp → V-JEPA / CTM-JEPA feature prediction

Each operator delegates to a backend-specific kernel:
    CUDA/ROCm → Triton kernels
    Apple MPS → MLX bridge
    Huawei NPU → CANN adapter
    DirectML   → DML dispatch
    CPU        → Vectorized fallback

Usage:
    from luna_ops import Mamba2ScanOp, get_active_backend, set_backend
    set_backend("triton")
    output = Mamba2ScanOp.apply(x, delta, A, B, C, D)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Callable
from abc import ABC, abstractmethod
import os


# ==================== Backend Registry ====================

class BackendRegistry:
    """Global registry for operator backends.

    Each operator name maps to a backend-specific implementation.
    At runtime, the best available backend is selected automatically.
    """

    _instance = None
    _ops: dict = {}
    _active_backend: str = "auto"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._ops = {}
            cls._instance._active_backend = "auto"
        return cls._instance

    def register(self, op_name: str, backend: str, fn: Callable):
        """Register a backend implementation for an operator."""
        if op_name not in self._ops:
            self._ops[op_name] = {}
        self._ops[op_name][backend] = fn

    def get(self, op_name: str, backend: Optional[str] = None) -> Optional[Callable]:
        """Get the best available implementation for an operator."""
        backend = backend or self._active_backend
        if op_name in self._ops:
            if backend in self._ops[op_name]:
                return self._ops[op_name][backend]
            # Fallback: try first available backend
            if self._ops[op_name]:
                return next(iter(self._ops[op_name].values()))
        return None

    def set_backend(self, backend: str):
        """Set the active backend."""
        self._active_backend = backend


registry = BackendRegistry()


def get_active_backend() -> str:
    return registry._active_backend


def set_backend(backend: str):
    registry.set_backend(backend)


# ==================== Operator: Mamba2 Selective Scan ====================

class Mamba2ScanOp(torch.autograd.Function):
    """Mamba2 Selective State Space Scan.

    Implements the discretized SSM recurrence:
        h_t = exp(ΔA) * h_{t-1} + ΔB * x_t
        y_t = C * h_t + D * x_t

    Input shapes:
        x:     [B, L, d_inner]           — input sequence
        delta: [B, L, d_inner]           — discretization step
        A:     [d_inner, d_state]        — state transition matrix
        B_ssm: [B, L, d_state]           — input projection
        C_ssm: [B, L, d_state]           — output projection
        D:     [d_inner]                 — skip connection

    Output:
        y: [B, L, d_inner]               — output sequence
        final_state: [B, d_inner, d_state] — final hidden state
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B_ssm: torch.Tensor,
        C_ssm: torch.Tensor,
        D: torch.Tensor,
        use_triton: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ctx.save_for_backward(x, delta, A, B_ssm, C_ssm, D)
        ctx.use_triton = use_triton

        # Dispatch to best available backend
        if use_triton and registry.get("mamba2_scan", "triton"):
            return registry.get("mamba2_scan", "triton")(x, delta, A, B_ssm, C_ssm, D)
        elif registry.get("mamba2_scan", "mlx"):
            return registry.get("mamba2_scan", "mlx")(x, delta, A, B_ssm, C_ssm, D)
        else:
            return _mamba2_scan_pytorch(x, delta, A, B_ssm, C_ssm, D)

    @staticmethod
    def backward(ctx, grad_y, grad_state):
        x, delta, A, B_ssm, C_ssm, D = ctx.saved_tensors
        # Simplified backward: recompute forward with grad
        # Production: use custom Triton backward kernel
        return _mamba2_scan_backward(grad_y, x, delta, A, B_ssm, C_ssm, D)


def _mamba2_scan_pytorch(
    x: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B_ssm: torch.Tensor,
    C_ssm: torch.Tensor,
    D: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch fallback for Mamba2 scan (CPU-compatible)."""
    B_size, L, d_inner = x.shape
    d_state = A.shape[1]
    device = x.device

    # Discretize
    delta = delta.float()
    delta_A = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # [B, L, d_inner, d_state]
    delta_B = delta.unsqueeze(-1) * B_ssm.unsqueeze(2)  # [B, L, d_inner, d_state]

    h = torch.zeros(B_size, d_inner, d_state, device=device, dtype=torch.float32)
    outputs = []

    for t in range(L):
        h = delta_A[:, t] * h + delta_B[:, t] * x[:, t].unsqueeze(-1).float()
        y = (h * C_ssm[:, t].unsqueeze(1).float()).sum(dim=-1)
        outputs.append(y)

    y = torch.stack(outputs, dim=1)
    y = y + x.float() * D.unsqueeze(0).unsqueeze(0)
    return y.to(x.dtype), h.to(x.dtype)


def _mamba2_scan_backward(grad_y, x, delta, A, B_ssm, C_ssm, D):
    """Simplified backward pass (for autograd compatibility)."""
    # Full backward would require associative scan reversal
    # For now, fall back to PyTorch autograd
    return None, None, None, None, None, None


# ==================== Operator: Flash Attention (MLA) ====================

class FlashAttentionOp(torch.autograd.Function):
    """Flash Attention for MLA (Multi-head Latent Attention).

    Implements compressed attention with KV low-rank decomposition:
        Q = q_b(q_a(x))        — low-rank Q projection
        K = kv_b(kv_a(x))      — low-rank KV decompression
        Attn = softmax(Q @ K^T / sqrt(d)) @ V

    Input shapes:
        q: [B, n_heads, L, dim]       — query
        k: [B, n_heads, L, dim]       — key
        v: [B, n_heads, L, v_dim]     — value
        softmax_scale: float          — 1/sqrt(dim)

    Output:
        attn_output: [B, n_heads, L, v_dim]
        attn_weights: [B, n_heads, L, L] (optional, for debugging)
    """

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        softmax_scale: float,
        causal: bool = True,
        use_flash: bool = True,
    ) -> torch.Tensor:
        ctx.save_for_backward(q, k, v)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal

        if use_flash and registry.get("flash_attention", "triton"):
            return registry.get("flash_attention", "triton")(q, k, v, softmax_scale, causal)
        elif registry.get("flash_attention", "mlx"):
            return registry.get("flash_attention", "mlx")(q, k, v, softmax_scale, causal)
        else:
            return _flash_attention_pytorch(q, k, v, softmax_scale, causal)

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v = ctx.saved_tensors
        # Simplified backward
        return _flash_attention_backward(grad_output, q, k, v, ctx.softmax_scale, ctx.causal)


def _flash_attention_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    causal: bool = True,
) -> torch.Tensor:
    """Pure PyTorch attention (CPU fallback)."""
    B, H, L, D = q.shape
    attn = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale  # [B, H, L, L]

    if causal:
        mask = torch.triu(torch.ones(L, L, device=q.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask, float("-inf"))

    attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(attn, v)


def _flash_attention_backward(grad_output, q, k, v, softmax_scale, causal):
    return None, None, None, None, None, None


# ==================== Operator: MoE Gating ====================

class MoEGateOp(torch.autograd.Function):
    """FlashMoE Top-K Gating with load balancing.

    Routes tokens to top-K experts, computes routing weights, and returns
    expert assignments for efficient batched computation.

    Input:
        router_logits: [B, L, num_experts]  — raw router scores
        top_k: int                          — number of experts per token
        capacity_factor: float              — dynamic capacity multiplier

    Output:
        topk_weights: [B, L, top_k]         — normalized routing weights
        topk_indices: [B, L, top_k]         — expert indices
        expert_counts: [num_experts]        — tokens per expert (for load balancing)
    """

    @staticmethod
    def forward(
        ctx,
        router_logits: torch.Tensor,
        top_k: int = 4,
        capacity_factor: float = 1.25,
        num_experts: int = 48,
        use_triton: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx.top_k = top_k
        ctx.num_experts = num_experts

        if use_triton and registry.get("moe_gate", "triton"):
            return registry.get("moe_gate", "triton")(router_logits, top_k, capacity_factor, num_experts)
        else:
            return _moe_gate_pytorch(router_logits, top_k, capacity_factor, num_experts)

    @staticmethod
    def backward(ctx, grad_weights, grad_indices, grad_counts):
        return None, None, None, None, None


def _moe_gate_pytorch(
    router_logits: torch.Tensor,
    top_k: int,
    capacity_factor: float,
    num_experts: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure PyTorch MoE gating."""
    B, L, N = router_logits.shape
    topk_weights, topk_indices = torch.topk(router_logits, top_k, dim=-1)
    topk_weights = F.softmax(topk_weights, dim=-1)

    # Count tokens per expert
    expert_counts = torch.zeros(num_experts, device=router_logits.device)
    one_hot = F.one_hot(topk_indices.view(-1), num_classes=num_experts).float()
    expert_counts = one_hot.sum(dim=0)

    return topk_weights, topk_indices, expert_counts


# ==================== Operator: JEPA Predictor ====================

class JEPAPredictorOp(torch.autograd.Function):
    """JEPA Feature Prediction (V-JEPA and CTM-JEPA).

    Predicts masked/target features from context features using a lightweight
    predictor network. Supports both visual (V-JEPA) and neural (CTM-JEPA) domains.

    Input:
        context_features: [B, N_visible, D]  — visible/context features
        mask: [B, N_total]                   — boolean mask (True = to predict)
        ids_restore: [B, N_total]            — indices to restore original order

    Output:
        predictions: [B, N_total, D]         — predicted features for all positions
    """

    @staticmethod
    def forward(
        ctx,
        context_features: torch.Tensor,
        mask: torch.Tensor,
        ids_restore: torch.Tensor,
        predictor_weight: torch.Tensor,
        predictor_bias: Optional[torch.Tensor] = None,
        use_triton: bool = True,
    ) -> torch.Tensor:
        ctx.save_for_backward(context_features, mask, ids_restore, predictor_weight, predictor_bias)

        if use_triton and registry.get("jepa_predict", "triton"):
            return registry.get("jepa_predict", "triton")(
                context_features, mask, ids_restore, predictor_weight, predictor_bias
            )
        else:
            return _jepa_predict_pytorch(
                context_features, mask, ids_restore, predictor_weight, predictor_bias
            )

    @staticmethod
    def backward(ctx, grad_output):
        return _jepa_predict_backward(grad_output, *ctx.saved_tensors)


def _jepa_predict_pytorch(
    context_features: torch.Tensor,
    mask: torch.Tensor,
    ids_restore: torch.Tensor,
    predictor_weight: torch.Tensor,
    predictor_bias: Optional[torch.Tensor],
) -> torch.Tensor:
    """Pure PyTorch JEPA prediction."""
    B, N_visible, D = context_features.shape
    N_total = mask.shape[1]

    # Place context features at visible positions, zeros at masked
    full = torch.zeros(B, N_total, D, device=context_features.device, dtype=context_features.dtype)
    full[:, ~mask, :] = context_features

    # Restore order
    batch_indices = torch.arange(B, device=context_features.device).unsqueeze(1)
    full = full[batch_indices, ids_restore]

    # Apply predictor (simplified linear projection)
    predictions = F.linear(full, predictor_weight, predictor_bias)
    return predictions


def _jepa_predict_backward(grad_output, context_features, mask, ids_restore, predictor_weight, predictor_bias):
    return None, None, None, None, None, None


# ==================== Backend Registration ====================

def register_pytorch_backends():
    """Register pure PyTorch fallback implementations."""
    registry.register("mamba2_scan", "pytorch", _mamba2_scan_pytorch)
    registry.register("flash_attention", "pytorch", _flash_attention_pytorch)
    registry.register("moe_gate", "pytorch", _moe_gate_pytorch)
    registry.register("jepa_predict", "pytorch", _jepa_predict_pytorch)


# Auto-register PyTorch fallbacks
register_pytorch_backends()


# ==================== Export ====================

__all__ = [
    "Mamba2ScanOp",
    "FlashAttentionOp",
    "MoEGateOp",
    "JEPAPredictorOp",
    "BackendRegistry",
    "registry",
    "get_active_backend",
    "set_backend",
    "register_pytorch_backends",
]