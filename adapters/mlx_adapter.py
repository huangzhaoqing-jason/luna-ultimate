"""MLX Backend Adapter for Apple Silicon.

Maps Luna-Ultimate operators to Apple MLX framework primitives.
MLX provides unified memory and Metal-accelerated computation for M1/M2/M3/M4.

Key mappings:
  - Mamba2 scan → mlx.core.fast.ssd (or manual scan with mlx operations)
  - Flash Attention → mlx.nn.fast.scaled_dot_product_attention
  - MoE gating → mlx top-k + softmax (native)
  - JEPA → mlx linear projections

Requirements:
    pip install mlx

Usage:
    from adapters.mlx_adapter import register_mlx_backends
    register_mlx_backends()
"""

import math
from typing import Tuple, Optional

try:
    import mlx.core as mx
    import mlx.nn as mlx_nn
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


def _ensure_mlx_tensor(x) -> "mx.array":
    """Convert PyTorch tensor to MLX array if needed."""
    if HAS_MLX and not isinstance(x, mx.array):
        return mx.array(x.cpu().numpy())
    return x


def _ensure_torch_tensor(x, original_device=None):
    """Convert MLX array back to PyTorch tensor."""
    import torch
    if HAS_MLX and isinstance(x, mx.array):
        t = torch.from_numpy(x.__array__())
        if original_device is not None:
            t = t.to(original_device)
        return t
    return x


def mamba2_scan_mlx(
    x, delta, A, B_ssm, C_ssm, D,
) -> Tuple:
    """Mamba2 selective scan via MLX.

    MLX doesn't have a native SSM scan kernel, so we implement it
    using MLX's vectorized operations. For production, this should
    be replaced with a custom Metal shader via mlx.fast.

    Input:  x [B, L, d_inner], delta [B, L, d_inner], A [d_inner, d_state],
            B [B, L, d_state], C [B, L, d_state], D [d_inner]
    Output: y [B, L, d_inner], final_state [B, d_inner, d_state]
    """
    if not HAS_MLX:
        raise RuntimeError("MLX not available. Install: pip install mlx")

    import torch
    dev = x.device if isinstance(x, torch.Tensor) else None

    # Convert to MLX
    x_mx = _ensure_mlx_tensor(x)
    delta_mx = _ensure_mlx_tensor(delta)
    A_mx = _ensure_mlx_tensor(A)
    B_mx = _ensure_mlx_tensor(B_ssm)
    C_mx = _ensure_mlx_tensor(C_ssm)
    D_mx = _ensure_mlx_tensor(D)

    B_size, L, d_inner = x_mx.shape
    d_state = A_mx.shape[1]

    # Discretize
    delta_float = mx.array(delta_mx, dtype=mx.float32)
    A_float = mx.array(A_mx, dtype=mx.float32)
    delta_A = mx.exp(delta_float[..., None] * A_float[None, None, :, :])
    delta_B = delta_float[..., None] * mx.array(B_mx, dtype=mx.float32)[..., None, :]

    # MLX-accelerated sequential scan
    h = mx.zeros((B_size, d_inner, d_state), dtype=mx.float32)
    outputs = []

    for t in range(L):
        h = delta_A[:, t] * h + delta_B[:, t] * mx.array(x_mx, dtype=mx.float32)[:, t, :, None]
        y_t = mx.sum(h * mx.array(C_mx, dtype=mx.float32)[:, t, None, :], axis=-1)
        y_t = y_t + mx.array(x_mx, dtype=mx.float32)[:, t] * mx.array(D_mx, dtype=mx.float32)[None, :]
        outputs.append(y_t)

    y_mx = mx.stack(outputs, axis=1)

    # Convert back
    y = _ensure_torch_tensor(y_mx, dev)
    final_state = _ensure_torch_tensor(h, dev)
    return y, final_state


def flash_attention_mlx(
    q, k, v, softmax_scale: float, causal: bool = True,
):
    """Flash Attention via MLX scaled_dot_product_attention.

    MLX's native attention is Metal-optimized and supports causal masking.

    Input:  q [B, H, L, D], k [B, H, L, D], v [B, H, L, D_v]
    Output: [B, H, L, D_v]
    """
    if not HAS_MLX:
        raise RuntimeError("MLX not available. Install: pip install mlx")

    import torch
    dev = q.device if isinstance(q, torch.Tensor) else None

    q_mx = _ensure_mlx_tensor(q)
    k_mx = _ensure_mlx_tensor(k)
    v_mx = _ensure_mlx_tensor(v)

    # MLX sdpa
    mask = None
    if causal:
        L = q_mx.shape[2]
        mask = mx.triu(mx.full((L, L), float("-inf")), k=1)

    attn_out = mx.fast.scaled_dot_product_attention(
        q_mx, k_mx, v_mx, scale=softmax_scale, mask=mask
    )

    return _ensure_torch_tensor(attn_out, dev)


def moe_gate_mlx(
    router_logits, top_k: int, capacity_factor: float, num_experts: int,
) -> Tuple:
    """MoE Top-K gating via MLX native operations.

    MLX provides efficient top-k and softmax on Apple Silicon.
    """
    if not HAS_MLX:
        raise RuntimeError("MLX not available. Install: pip install mlx")

    import torch
    dev = router_logits.device if isinstance(router_logits, torch.Tensor) else None

    logits_mx = _ensure_mlx_tensor(router_logits)
    B, L, N = logits_mx.shape

    # Top-K
    topk_vals, topk_indices = mx.topk(logits_mx, k=top_k, axis=-1)

    # Softmax over selected experts
    max_val = mx.max(topk_vals, axis=-1, keepdims=True)
    exp_vals = mx.exp(topk_vals - max_val)
    weights = exp_vals / mx.sum(exp_vals, axis=-1, keepdims=True)

    # Expert counts
    flat_indices = topk_indices.reshape(-1)
    counts = mx.zeros(num_experts, dtype=mx.int32)
    for i in range(flat_indices.shape[0]):
        e = flat_indices[i].item()
        counts = counts.at[e].add(1)

    return (
        _ensure_torch_tensor(weights, dev),
        _ensure_torch_tensor(topk_indices, dev),
        _ensure_torch_tensor(counts, dev),
    )


def register_mlx_backends():
    """Register MLX backend implementations."""
    from luna_ops import registry

    if HAS_MLX:
        registry.register("mamba2_scan", "mlx", mamba2_scan_mlx)
        registry.register("flash_attention", "mlx", flash_attention_mlx)
        registry.register("moe_gate", "mlx", moe_gate_mlx)
        return True
    return False


__all__ = [
    "mamba2_scan_mlx",
    "flash_attention_mlx",
    "moe_gate_mlx",
    "register_mlx_backends",
    "HAS_MLX",
]