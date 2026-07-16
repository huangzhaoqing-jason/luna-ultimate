"""CANN Backend Adapter for Huawei Ascend NPU.

Maps Luna-Ultimate operators to Huawei Ascend C (CANN) primitives.
Uses torch_npu bridge for PyTorch integration and torch_npu.npu_ops for
Ascend-optimized kernels.

Key operator mapping table:
    Luna Op              → Ascend C Kernel
    ─────────────────────────────────────────────
    Mamba2 Scan          → npu_mamba2_scan (custom Ascend C)
    Flash Attention MLA  → npu_fusion_attention (Ascend ATB)
    MoE Top-K Gate       → npu_topk + npu_softmax
    JEPA Predict         → npu_matmul + npu_add

Requirements:
    - Huawei Ascend NPU (910B or later)
    - CANN toolkit (Ascend-cann-toolkit)
    - torch_npu (pip install torch-npu)
    - MindSpore (optional, for Ascend C kernel compilation)

Usage:
    from adapters.cann_adapter import register_cann_backends
    register_cann_backends()
"""

import torch
from typing import Tuple, Optional

try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    HAS_CANN = True
except ImportError:
    HAS_CANN = False


# ==================== CANN Operator Mappings ====================

def mamba2_scan_cann(
    x: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B_ssm: torch.Tensor,
    C_ssm: torch.Tensor,
    D: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mamba2 scan via CANN-optimized operations.

    Maps to Ascend C primitives:
      - exp → aclnnExp (Ascend optimized)
      - matmul → aclMatMul (Cube unit accelerated)
      - element-wise → aclnnMul, aclnnAdd

    For production, this should use a custom Ascend C kernel compiled
    with the CANN toolchain for optimal performance.
    """
    if not HAS_CANN:
        raise RuntimeError("CANN not available. Install torch-npu and CANN toolkit.")

    B, L, d_inner = x.shape
    d_state = A.shape[1]
    device = x.device

    # Ensure tensors are on NPU
    if "npu" not in str(device):
        x = x.to("npu:0")
        delta = delta.to("npu:0")
        A = A.to("npu:0")
        B_ssm = B_ssm.to("npu:0")
        C_ssm = C_ssm.to("npu:0")
        D = D.to("npu:0")

    # Discretize (Ascend-optimized)
    delta_f = delta.float()
    delta_A = torch.exp(delta_f.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
    delta_B = delta_f.unsqueeze(-1) * B_ssm.unsqueeze(2)

    # Sequential scan (replace with custom Ascend C kernel for production)
    h = torch.zeros(B, d_inner, d_state, device=device, dtype=torch.float32)
    outputs = []

    for t in range(L):
        # Use Ascend-optimized element-wise ops
        h = delta_A[:, t] * h + delta_B[:, t] * x[:, t].unsqueeze(-1).float()
        y = (h * C_ssm[:, t].unsqueeze(1).float()).sum(dim=-1)
        outputs.append(y)

    y = torch.stack(outputs, dim=1)
    y = y + x.float() * D.unsqueeze(0).unsqueeze(0)

    return y.to(x.dtype), h.to(x.dtype)


def flash_attention_cann(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    causal: bool = True,
) -> torch.Tensor:
    """Flash Attention via CANN Fusion Attention.

    Uses npu_fusion_attention when available (Ascend ATB),
    falling back to standard PyTorch attention with NPU optimization.
    """
    if not HAS_CANN:
        raise RuntimeError("CANN not available.")

    # Ensure NPU tensors
    if "npu" not in str(q.device):
        q = q.to("npu:0")
        k = k.to("npu:0")
        v = v.to("npu:0")

    # Try Ascend fusion attention
    try:
        from torch_npu.npu import npu_fusion_attention
        attn_out = npu_fusion_attention(
            q, k, v, head_num=q.shape[1],
            input_layout="BNSD", scale=softmax_scale,
        )
        return attn_out
    except (ImportError, AttributeError):
        pass

    # Fallback: PyTorch attention on NPU
    B, H, L, D = q.shape
    attn = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale

    if causal:
        mask = torch.triu(
            torch.ones(L, L, device=q.device, dtype=torch.bool), diagonal=1
        )
        attn = attn.masked_fill(mask, float("-inf"))

    attn = torch.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(attn, v)


def moe_gate_cann(
    router_logits: torch.Tensor,
    top_k: int,
    capacity_factor: float,
    num_experts: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MoE gating via CANN-optimized top-k.

    Ascend NPU has strong matrix capabilities — leverage Cube units
    for large expert count scenarios.
    """
    if not HAS_CANN:
        raise RuntimeError("CANN not available.")

    if "npu" not in str(router_logits.device):
        router_logits = router_logits.to("npu:0")

    topk_weights, topk_indices = torch.topk(router_logits, top_k, dim=-1)
    topk_weights = torch.softmax(topk_weights, dim=-1)

    expert_counts = torch.zeros(num_experts, device=router_logits.device)
    one_hot = torch.nn.functional.one_hot(
        topk_indices.view(-1), num_classes=num_experts
    ).float()
    expert_counts = one_hot.sum(dim=0)

    return topk_weights, topk_indices, expert_counts


def register_cann_backends():
    """Register CANN backend implementations."""
    from luna_ops import registry

    if HAS_CANN:
        registry.register("mamba2_scan", "cann", mamba2_scan_cann)
        registry.register("flash_attention", "cann", flash_attention_cann)
        registry.register("moe_gate", "cann", moe_gate_cann)
        return True
    return False


def smoke_probe() -> dict:
    """CANN 冒烟：有 torch_npu 则探测；否则诚实报告 CPU 回退。"""
    import torch
    if not HAS_CANN:
        return {
            "ok": True,
            "available": False,
            "detail": "torch_npu/CANN not installed; luna serve --device cann → cpu fallback",
        }
    try:
        x = torch.randn(1, 4, 8)
        if hasattr(torch, "npu") and torch.npu.is_available():
            x = x.to("npu:0")
            y = x + 1
            return {"ok": True, "available": True, "device": str(y.device)}
        return {"ok": True, "available": True, "detail": "torch_npu imported; npu not available"}
    except Exception as e:
        return {"ok": False, "available": True, "error": str(e)}


__all__ = [
    "mamba2_scan_cann",
    "flash_attention_cann",
    "moe_gate_cann",
    "register_cann_backends",
    "smoke_probe",
    "HAS_CANN",
]