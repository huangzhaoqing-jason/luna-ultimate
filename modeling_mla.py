"""Multi-head Latent Attention (MLA) for Luna-Ultimate (Layers 13-32).

Implements MLA with low-rank Q/KV compression, decoupled RoPE, and YaRN extension.
Based on DeepSeek-V2 architecture.

Key dimensions:
  - d_model = 8192
  - n_heads = 32
  - qk_nope_head_dim = 128 (non-RoPE Q/K dim per head)
  - qk_rope_head_dim = 64  (RoPE Q/K dim per head)
  - v_head_dim = 128
  - kv_lora_rank = 1024 (KV compression bottleneck)
  - q_lora_rank = 3072   (Q compression bottleneck)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math
from config import LunaConfig


class YaRNRotaryEmbedding(nn.Module):
    """YaRN-extended Rotary Position Embedding.

    Extends RoPE to 128K via YaRN with factor 8.0 and base theta 10000.
    """

    def __init__(self, config: LunaConfig):
        super().__init__()
        self.dim = config.qk_rope_head_dim  # 64
        self.max_seq_len = config.max_position_embeddings  # 131072
        self.theta = config.rope_theta  # 10000
        self.yarn_factor = config.yarn_factor  # 8.0

        self._build_freqs()

    def _build_freqs(self):
        """Build YaRN frequency cache."""
        dim_half = self.dim // 2
        freqs = 1.0 / (
            self.theta ** (torch.arange(0, dim_half, dtype=torch.float32) / dim_half)
        )
        # YaRN scaling: adjust frequencies
        freqs = freqs / self.yarn_factor
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute cos/sin for positions 0..seq_len-1.

        Returns:
            cos: [seq_len, dim]
            sin: [seq_len, dim]
        """
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        # [seq_len, 1] × [1, dim_half] -> [seq_len, dim_half]
        angles = positions.unsqueeze(1) * self.freqs.unsqueeze(0)
        # Repeat for full dim
        angles = torch.cat([angles, angles], dim=-1)  # [seq_len, dim]
        return angles.cos(), angles.sin()


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary embedding to x.

    Args:
        x: [B, n_heads, L, dim] or [B, L, n_heads, dim]
        cos: [L, dim]
        sin: [L, dim]

    Returns:
        rotated: same shape as x
    """
    # Ensure x is [B, n_heads, L, dim]
    if x.shape[1] != cos.shape[0] and x.shape[2] == cos.shape[0]:
        # [B, L, H, D] -> [B, H, L, D]
        x = x.transpose(1, 2)

    dim_half = x.shape[-1] // 2
    x1, x2 = x[..., :dim_half], x[..., dim_half:]

    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, L, dim]
    sin = sin.unsqueeze(0).unsqueeze(0)

    rotated = torch.cat([
        x1 * cos[..., :dim_half] - x2 * sin[..., :dim_half],
        x2 * cos[..., :dim_half] + x1 * sin[..., :dim_half],
    ], dim=-1)

    return rotated


class MLAAttention(nn.Module):
    """Multi-head Latent Attention with low-rank compression.

    Q projection: d_model -> q_lora_rank -> n_heads * (k_nope + k_rope)
    KV compression: d_model -> kv_lora_rank + n_heads*k_rope
    KV decompression: kv_lora_rank -> n_heads * (k_nope + v)
    """

    def __init__(self, config: LunaConfig):
        super().__init__()
        self.d_model = config.hidden_size           # 8192
        self.n_heads = config.n_heads                # 32
        self.qk_nope_dim = config.qk_nope_head_dim  # 128
        self.qk_rope_dim = config.qk_rope_head_dim  # 64
        self.v_head_dim = config.v_head_dim          # 128
        self.kv_lora_rank = config.kv_lora_rank      # 1024
        self.q_lora_rank = config.q_lora_rank        # 3072
        self.softmax_scale = self.qk_nope_dim ** -0.5

        # Q total per head: k_nope + k_rope = 128 + 64 = 192
        self.q_per_head = self.qk_nope_dim + self.qk_rope_dim  # 192
        self.q_total = self.n_heads * self.q_per_head          # 6144

        # Q compression: d_model -> q_lora_rank -> q_total
        # [8192] -> [3072] -> [6144]
        self.q_a_proj = nn.Linear(self.d_model, self.q_lora_rank, bias=False)
        self.q_a_norm = nn.RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(self.q_lora_rank, self.q_total, bias=False)

        # KV compression: d_model -> kv_lora_rank + n_heads*k_rope
        # [8192] -> [1024] + [32*64=2048] = [3072]
        self.kv_a_lora_dim = self.kv_lora_rank + self.n_heads * self.qk_rope_dim  # 1024+2048=3072
        self.kv_a_proj = nn.Linear(self.d_model, self.kv_a_lora_dim, bias=False)
        self.kv_a_norm = nn.RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)

        # KV decompression: kv_lora_rank -> n_heads*(k_nope + v)
        # [1024] -> [32 * (128+128) = 8192]
        self.kv_b_dim = self.n_heads * (self.qk_nope_dim + self.v_head_dim)  # 32*256=8192
        self.kv_b_proj = nn.Linear(self.kv_lora_rank, self.kv_b_dim, bias=False)

        # Output projection: n_heads * v_head_dim -> d_model
        # [32*128=4096] -> [8192]
        self.o_proj = nn.Linear(self.n_heads * self.v_head_dim, self.d_model, bias=False)

        # RoPE
        self.rope = YaRNRotaryEmbedding(config)

        self.norm = nn.RMSNorm(self.d_model, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_int4_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """MLA forward pass.

        Args:
            hidden_states: [B, L, d_model=8192]
            kv_cache: Optional (compressed_kv, k_pe) from previous steps.
            use_int4_cache: Whether to quantize KV cache to INT4.

        Returns:
            output: [B, L, d_model=8192]
            new_kv_cache: (compressed_kv, k_pe) for next step.
        """
        B, L, D = hidden_states.shape
        residual = hidden_states
        hidden_states = self.norm(hidden_states)

        # ========== Q Projection ==========
        # [B, L, 8192] -> [B, L, 3072]
        q_compressed = self.q_a_proj(hidden_states)
        q_compressed = self.q_a_norm(q_compressed)
        # [B, L, 3072] -> [B, L, 6144]
        q_all = self.q_b_proj(q_compressed)

        # Reshape: [B, L, 6144] -> [B, L, n_heads, 192] -> [B, n_heads, L, 192]
        q_all = q_all.view(B, L, self.n_heads, self.q_per_head)
        q_all = q_all.transpose(1, 2)  # [B, n_heads, L, 192]

        # Split into nope and rope parts
        q_nope = q_all[..., :self.qk_nope_dim]   # [B, n_heads, L, 128]
        q_rope = q_all[..., self.qk_nope_dim:]   # [B, n_heads, L, 64]

        # ========== KV Compression ==========
        # [B, L, 8192] -> [B, L, 3072] (1024 compressed + 2048 k_rope)
        kv_a = self.kv_a_proj(hidden_states)  # [B, L, 3072]

        # Split
        kv_compressed = kv_a[..., :self.kv_lora_rank]  # [B, L, 1024]
        k_rope = kv_a[..., self.kv_lora_rank:]          # [B, L, 2048]

        # Normalize compressed KV
        kv_compressed = self.kv_a_norm(kv_compressed)

        # ========== KV Decompression ==========
        # [B, L, 1024] -> [B, L, 8192]
        kv_decompressed = self.kv_b_proj(kv_compressed)
        # [B, L, 8192] -> [B, L, n_heads, 256] -> [B, n_heads, L, 256]
        kv_decompressed = kv_decompressed.view(B, L, self.n_heads, self.qk_nope_dim + self.v_head_dim)
        kv_decompressed = kv_decompressed.transpose(1, 2)

        k_nope = kv_decompressed[..., :self.qk_nope_dim]   # [B, n_heads, L, 128]
        v = kv_decompressed[..., self.qk_nope_dim:]         # [B, n_heads, L, 128]

        # Reshape k_rope: [B, L, 2048] -> [B, n_heads, L, 64]
        k_rope = k_rope.view(B, L, self.n_heads, self.qk_rope_dim)
        k_rope = k_rope.transpose(1, 2)  # [B, n_heads, L, 64]

        # ========== Apply RoPE (YaRN) ==========
        cos, sin = self.rope(L, hidden_states.device)
        # cos, sin: [L, 64]

        # Apply RoPE to q_rope and k_rope
        q_rope = apply_rotary_emb(q_rope, cos, sin)  # [B, n_heads, L, 64]
        k_rope = apply_rotary_emb(k_rope, cos, sin)  # [B, n_heads, L, 64]

        # ========== Concatenate Q and K ==========
        q = torch.cat([q_nope, q_rope], dim=-1)  # [B, n_heads, L, 192]
        k = torch.cat([k_nope, k_rope], dim=-1)  # [B, n_heads, L, 192]

        # ========== Attention ==========
        # [B, n_heads, L, 192] @ [B, n_heads, 192, L] -> [B, n_heads, L, L]
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.softmax_scale

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(L, L, device=hidden_states.device, dtype=torch.bool), diagonal=1
        )
        attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)

        # [B, n_heads, L, L] @ [B, n_heads, L, 128] -> [B, n_heads, L, 128]
        attn_output = torch.matmul(attn_weights, v)

        # ========== Output Projection ==========
        # [B, n_heads, L, 128] -> [B, L, 4096] -> [B, L, 8192]
        attn_output = attn_output.transpose(1, 2).contiguous()  # [B, L, n_heads, 128]
        attn_output = attn_output.view(B, L, self.n_heads * self.v_head_dim)  # [B, L, 4096]
        output = self.o_proj(attn_output)  # [B, L, 8192]

        # Residual
        output = output + residual

        # New KV cache: store compressed KV and k_rope
        new_kv_cache = (kv_compressed, k_rope)

        return output, new_kv_cache


class MLABlock(nn.Module):
    """MLA block with FlashMoE and CTM residual injection.

    Used in layers 13-32 of Luna-Ultimate.

    Args:
        config: LunaConfig.
        layer_idx: Layer index (12-31).
    """

    def __init__(self, config: LunaConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.attention = MLAAttention(config)
        self.moe_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        moe_layer: nn.Module,
        ctm_residual: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_int4_cache: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """MLA Block forward.

        Args:
            hidden_states: [B, L, d_model]
            moe_layer: FlashMoE module for this layer.
            ctm_residual: [B, L, d_model] from CTM.
            kv_cache: Previous KV cache.
            use_int4_cache: Whether to use INT4 KV cache.

        Returns:
            hidden_states: [B, L, d_model]
            moe_aux_loss: scalar auxiliary MoE loss.
            new_kv_cache: Updated KV cache.
        """
        # MLA attention
        hidden_states, new_kv_cache = self.attention(hidden_states, kv_cache, use_int4_cache)

        # CTM residual injection
        hidden_states = hidden_states + ctm_residual

        # FlashMoE
        normed = self.moe_norm(hidden_states)
        hidden_states, aux_loss = moe_layer(normed, hidden_states)

        return hidden_states, aux_loss, new_kv_cache