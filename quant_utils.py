"""Quantization utilities for Luna-Ultimate.

Provides INT4/INT8 simulated quantization hooks, RMSNorm, and SwiGLU activation.
All Linear layers support quantize/dequantize interfaces for precision-friendly
deployment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, L, D] or [B, D] -> [B, L, D] or [B, D]"""
        dtype = x.dtype
        x = x.float()
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * rms).to(dtype) * self.weight


class SwiGLU(nn.Module):
    """SwiGLU FFN: gate(x) * up(x) with SiLU gate."""

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


def simulate_quantize_int8(x: torch.Tensor) -> torch.Tensor:
    """Simulate INT8 quantization round-trip."""
    x_min = x.min().item()
    x_max = x.max().item()
    scale = (x_max - x_min) / 255.0
    if scale == 0:
        return x
    zero_point = round(-x_min / scale)
    zero_point = max(0, min(255, zero_point))
    x_q = torch.clamp(torch.round(x / scale + zero_point), 0, 255)
    return (x_q - zero_point) * scale


def simulate_quantize_int4(x: torch.Tensor) -> torch.Tensor:
    """Simulate INT4 quantization round-trip."""
    x_min = x.min().item()
    x_max = x.max().item()
    scale = (x_max - x_min) / 15.0
    if scale == 0:
        return x
    zero_point = round(-x_min / scale)
    zero_point = max(0, min(15, zero_point))
    x_q = torch.clamp(torch.round(x / scale + zero_point), 0, 15)
    return (x_q - zero_point) * scale


def pack_int4_to_int8(x_int4: torch.Tensor) -> torch.Tensor:
    """Pack two INT4 values into one INT8. [..., N] -> [..., N//2]"""
    x_int4 = x_int4.to(torch.uint8)
    even = x_int4[..., 0::2]
    odd = x_int4[..., 1::2]
    return (even << 4) | odd


def unpack_int8_to_int4(x_packed: torch.Tensor) -> torch.Tensor:
    """Unpack INT8 to INT4. [..., N] -> [..., 2*N]"""
    high = (x_packed >> 4) & 0x0F
    low = x_packed & 0x0F
    return torch.stack([high, low], dim=-1).flatten(-2)


def quantize_kv_cache_int4(
    kv_compressed: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize compressed KV cache (kv_lora_rank dim) to INT4.

    Args:
        kv_compressed: [B, n_heads, L, kv_lora_rank=1024]

    Returns:
        packed_kv: [B, n_heads, L, 512] - INT4 packed into INT8
        scale: [B, n_heads, 1, 1]
        kv_min: [B, n_heads, 1, 1]
    """
    kv_min = kv_compressed.min(dim=-1, keepdim=True)[0]
    kv_max = kv_compressed.max(dim=-1, keepdim=True)[0]
    scale = (kv_max - kv_min) / 15.0
    scale = torch.clamp(scale, min=1e-8)
    kv_int4 = torch.clamp(
        torch.round((kv_compressed - kv_min) / scale), 0, 15
    ).to(torch.uint8)
    packed = pack_int4_to_int8(kv_int4)
    return packed, scale, kv_min


def dequantize_kv_cache_int4(
    packed_kv: torch.Tensor,
    scale: torch.Tensor,
    kv_min: torch.Tensor,
) -> torch.Tensor:
    """Dequantize INT4-packed KV cache back to float.

    Returns: [B, n_heads, L, kv_lora_rank]
    """
    kv_int4 = unpack_int8_to_int4(packed_kv).float()
    return kv_int4 * scale + kv_min


class QuantizedLinear(nn.Module):
    """Linear layer with INT4/INT8 quantization hooks."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        bits: int = 8,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.bits = bits
        self.quantize_fn = simulate_quantize_int4 if bits == 4 else simulate_quantize_int8

    def forward(self, x: torch.Tensor, quantize: bool = False) -> torch.Tensor:
        """[..., in_features] -> [..., out_features]"""
        weight = self.quantize_fn(self.linear.weight) if quantize else self.linear.weight
        return F.linear(x, weight, self.linear.bias)

    def get_quantized_weight(self) -> torch.Tensor:
        """Return quantized weight."""
        return self.quantize_fn(self.linear.weight)


class PagedKVCache:
    """PagedAttention-compatible KV cache for MLA layers.

    Stores compressed KV in INT4-packed pages for efficient memory management.
    Each page holds `page_size` tokens of compressed KV.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        kv_lora_rank: int,
        max_batch_size: int,
        max_seq_len: int,
        page_size: int = 256,
        dtype: torch.dtype = torch.float16,
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.kv_lora_rank = kv_lora_rank
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.page_size = page_size
        self.num_pages = (max_seq_len + page_size - 1) // page_size

        # Packed KV: kv_lora_rank // 2 (since 2 INT4 per INT8)
        self.packed_dim = kv_lora_rank // 2

        # [num_layers, num_pages, max_batch_size, num_heads, page_size, packed_dim]
        self.kv_cache = torch.zeros(
            num_layers,
            self.num_pages,
            max_batch_size,
            num_heads,
            page_size,
            self.packed_dim,
            dtype=torch.uint8,
        )
        self.scales = torch.zeros(
            num_layers, max_batch_size, num_heads, self.num_pages, 1, dtype=dtype
        )
        self.kv_mins = torch.zeros(
            num_layers, max_batch_size, num_heads, self.num_pages, 1, dtype=dtype
        )

        # Page table: [batch_size, max_seq_len // page_size] -> page index
        self.page_table = torch.full(
            (max_batch_size, self.num_pages), -1, dtype=torch.int32
        )
        self.num_allocated = torch.zeros(max_batch_size, dtype=torch.int32)
        self.free_pages = list(range(self.num_pages - 1, -1, -1))

    def allocate_page(self, batch_idx: int) -> int:
        """Allocate a new page for a batch element."""
        if not self.free_pages:
            raise RuntimeError("No free pages in KV cache")
        page_idx = self.free_pages.pop()
        self.page_table[batch_idx, self.num_allocated[batch_idx]] = page_idx
        self.num_allocated[batch_idx] += 1
        return page_idx

    def store(
        self,
        layer_idx: int,
        batch_idx: int,
        token_pos: int,
        kv_compressed: torch.Tensor,
        k_pe: torch.Tensor,
    ):
        """Store compressed KV for one token.

        Args:
            layer_idx: Which layer.
            batch_idx: Which batch element.
            token_pos: Position in the sequence.
            kv_compressed: [n_heads, kv_lora_rank] compressed KV.
            k_pe: [n_heads, k_rope_dim] RoPE key (not compressed).
        """
        page_idx = token_pos // self.page_size
        offset = token_pos % self.page_size
        actual_page = self.page_table[batch_idx, page_idx]
        if actual_page < 0:
            actual_page = self.allocate_page(batch_idx)

        packed, scale, kv_min = quantize_kv_cache_int4(
            kv_compressed.unsqueeze(0).unsqueeze(0)
        )
        self.kv_cache[layer_idx, actual_page, batch_idx, :, offset, :] = packed[0, 0]
        self.scales[layer_idx, batch_idx, :, actual_page, :] = scale[0, :, 0, :]
        self.kv_mins[layer_idx, batch_idx, :, actual_page, :] = kv_min[0, :, 0, :]
        return k_pe

    def retrieve(
        self,
        layer_idx: int,
        batch_idx: int,
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve compressed KV for a sequence.

        Returns:
            kv_compressed: [n_heads, seq_len, kv_lora_rank]
            k_pe: [n_heads, seq_len, k_rope_dim]
        """
        num_pages_needed = (seq_len + self.page_size - 1) // self.page_size
        kv_list = []
        for p in range(num_pages_needed):
            actual_page = self.page_table[batch_idx, p]
            page_kv = self.kv_cache[layer_idx, actual_page, batch_idx]
            page_scale = self.scales[layer_idx, batch_idx, :, actual_page, :]
            page_min = self.kv_mins[layer_idx, batch_idx, :, actual_page, :]
            page_kv_float = dequantize_kv_cache_int4(page_kv, page_scale, page_min)
            kv_list.append(page_kv_float)

        kv_compressed = torch.cat(kv_list, dim=1)[:, :seq_len, :]
        return kv_compressed, None