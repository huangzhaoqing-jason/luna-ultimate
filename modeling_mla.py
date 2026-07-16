"""Multi-head Latent Attention (MLA) for Luna Evolve."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import LunaConfig
from quant_utils import (
    RMSNorm,
    dequantize_kv_cache_int4,
    quantize_kv_cache_int4,
)


class YaRNRotaryEmbedding(nn.Module):
    """YaRN-extended Rotary Position Embedding."""

    def __init__(self, config: LunaConfig):
        super().__init__()
        self.dim = config.qk_rope_head_dim
        self.max_seq_len = config.max_position_embeddings
        self.theta = config.rope_theta
        self.yarn_factor = config.yarn_factor
        self._build_freqs()

    def _build_freqs(self):
        dim_half = self.dim // 2
        freqs = 1.0 / (
            self.theta ** (torch.arange(0, dim_half, dtype=torch.float32) / dim_half)
        )
        freqs = freqs / self.yarn_factor
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(
        self, seq_len: int, device: torch.device, offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(offset, offset + seq_len, device=device, dtype=torch.float32)
        angles = positions.unsqueeze(1) * self.freqs.unsqueeze(0)
        angles = torch.cat([angles, angles], dim=-1)
        return angles.cos(), angles.sin()


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply RoPE. Prefers x as [B, n_heads, L, dim]; cos/sin as [L, dim]."""
    # Only transpose when clearly [B, L, H, D]
    if x.dim() == 4 and x.shape[1] == cos.shape[0] and x.shape[2] != cos.shape[0]:
        x = x.transpose(1, 2)

    dim_half = x.shape[-1] // 2
    x1, x2 = x[..., :dim_half], x[..., dim_half:]
    # Broadcast over batch and heads: [1, 1, L, dim]
    cos = cos.view(1, 1, cos.shape[0], cos.shape[1])
    sin = sin.view(1, 1, sin.shape[0], sin.shape[1])
    return torch.cat([
        x1 * cos[..., :dim_half] - x2 * sin[..., :dim_half],
        x2 * cos[..., :dim_half] + x1 * sin[..., :dim_half],
    ], dim=-1)


class MLAAttention(nn.Module):
    """MLA with KV cache concat and optional INT4 compressed cache."""

    def __init__(self, config: LunaConfig):
        super().__init__()
        self.d_model = config.hidden_size
        self.n_heads = config.n_heads
        self.qk_nope_dim = config.qk_nope_head_dim
        self.qk_rope_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.q_lora_rank = config.q_lora_rank
        self.softmax_scale = self.qk_nope_dim ** -0.5

        self.q_per_head = self.qk_nope_dim + self.qk_rope_dim
        self.q_total = self.n_heads * self.q_per_head

        self.q_a_proj = nn.Linear(self.d_model, self.q_lora_rank, bias=False)
        self.q_a_norm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(self.q_lora_rank, self.q_total, bias=False)

        self.kv_a_lora_dim = self.kv_lora_rank + self.n_heads * self.qk_rope_dim
        self.kv_a_proj = nn.Linear(self.d_model, self.kv_a_lora_dim, bias=False)
        self.kv_a_norm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)

        self.kv_b_dim = self.n_heads * (self.qk_nope_dim + self.v_head_dim)
        self.kv_b_proj = nn.Linear(self.kv_lora_rank, self.kv_b_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.v_head_dim, self.d_model, bias=False)
        self.rope = YaRNRotaryEmbedding(config)
        self.norm = RMSNorm(self.d_model, eps=config.rms_norm_eps)

    def _unpack_cache(
        self,
        kv_cache: Optional[Tuple],
        use_int4_cache: bool,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int]:
        if kv_cache is None:
            return None, None, 0

        if use_int4_cache and len(kv_cache) == 4:
            packed, scale, kv_min, k_pe = kv_cache
            # packed is [B, L, rank] packed along last dim
            kv_compressed = dequantize_kv_cache_int4(
                packed.unsqueeze(1), scale.unsqueeze(1), kv_min.unsqueeze(1)
            ).squeeze(1)
            return kv_compressed, k_pe, kv_compressed.shape[1]

        kv_compressed, k_pe = kv_cache[0], kv_cache[1]
        return kv_compressed, k_pe, kv_compressed.shape[1]

    def _pack_cache(
        self,
        kv_compressed: torch.Tensor,
        k_rope: torch.Tensor,
        use_int4_cache: bool,
    ) -> Tuple:
        if use_int4_cache:
            # quantize_kv_cache_int4 expects [B, n_heads, L, rank] — use fake head dim=1
            packed, scale, kv_min = quantize_kv_cache_int4(
                kv_compressed.unsqueeze(1)
            )
            return (
                packed.squeeze(1),
                scale.squeeze(1),
                kv_min.squeeze(1),
                k_rope.detach(),
            )
        return (kv_compressed, k_rope)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: Optional[Tuple] = None,
        use_int4_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        B, L, D = hidden_states.shape
        residual = hidden_states
        hidden_states = self.norm(hidden_states)

        q_compressed = self.q_a_norm(self.q_a_proj(hidden_states))
        q_all = self.q_b_proj(q_compressed)
        q_all = q_all.view(B, L, self.n_heads, self.q_per_head).transpose(1, 2)
        q_nope = q_all[..., : self.qk_nope_dim]
        q_rope = q_all[..., self.qk_nope_dim :]

        kv_a = self.kv_a_proj(hidden_states)
        kv_compressed_new = self.kv_a_norm(kv_a[..., : self.kv_lora_rank])
        k_rope_raw = kv_a[..., self.kv_lora_rank :]

        past_kv, past_k_pe, past_len = self._unpack_cache(kv_cache, use_int4_cache)

        # RoPE with offset for cached positions
        cos, sin = self.rope(L, hidden_states.device, offset=past_len)
        k_rope = k_rope_raw.view(B, L, self.n_heads, self.qk_rope_dim).transpose(1, 2)
        q_rope = apply_rotary_emb(q_rope, cos, sin)
        k_rope = apply_rotary_emb(k_rope, cos, sin)

        if past_kv is not None:
            kv_compressed = torch.cat([past_kv, kv_compressed_new], dim=1)
            k_rope_full = torch.cat([past_k_pe, k_rope], dim=2)
        else:
            kv_compressed = kv_compressed_new
            k_rope_full = k_rope

        kv_len = kv_compressed.shape[1]
        kv_decompressed = self.kv_b_proj(kv_compressed)
        kv_decompressed = kv_decompressed.view(
            B, kv_len, self.n_heads, self.qk_nope_dim + self.v_head_dim
        ).transpose(1, 2)
        k_nope = kv_decompressed[..., : self.qk_nope_dim]
        v = kv_decompressed[..., self.qk_nope_dim :]

        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat([k_nope, k_rope_full], dim=-1)

        # Prefer runtime flash attention when available (inference)
        if not torch.is_grad_enabled():
            try:
                from runtime_manager import get_runtime
                rt = get_runtime(verbose=False)
                attn_output = rt.flash_attention(
                    q, k, v, self.softmax_scale, causal=True
                )
            except Exception:
                attn_output = self._eager_attention(q, k, v, L, kv_len, hidden_states.device)
        else:
            attn_output = self._eager_attention(q, k, v, L, kv_len, hidden_states.device)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(B, L, self.n_heads * self.v_head_dim)
        output = self.o_proj(attn_output) + residual

        new_kv_cache = self._pack_cache(kv_compressed, k_rope_full, use_int4_cache)
        return output, new_kv_cache

    def _eager_attention(self, q, k, v, q_len, kv_len, device):
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.softmax_scale
        # Causal mask aligned to past+current
        # query positions: past_len .. past_len+q_len-1
        # key positions: 0 .. kv_len-1
        past_len = kv_len - q_len
        q_pos = torch.arange(past_len, past_len + q_len, device=device).unsqueeze(1)
        k_pos = torch.arange(kv_len, device=device).unsqueeze(0)
        causal_mask = k_pos > q_pos
        attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        return torch.matmul(attn_weights, v)


class MLABlock(nn.Module):
    """MLA block with FlashMoE and CTM residual injection."""

    def __init__(self, config: LunaConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.attention = MLAAttention(config)
        self.moe_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        moe_layer: nn.Module,
        ctm_residual: torch.Tensor,
        kv_cache: Optional[Tuple] = None,
        use_int4_cache: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple]]:
        hidden_states, new_kv_cache = self.attention(
            hidden_states, kv_cache, use_int4_cache
        )
        hidden_states = hidden_states + ctm_residual
        normed = self.moe_norm(hidden_states)
        hidden_states, aux_loss = moe_layer(normed, hidden_states)
        return hidden_states, aux_loss, new_kv_cache
