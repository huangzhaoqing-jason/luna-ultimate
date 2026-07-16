"""Luna Evolve cost model: FLOPs, memory, and $/token estimates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from config import LunaConfig, count_parameters


@dataclass
class HardwarePrice:
    """Placeholder unit economics for cost-normalized comparisons."""

    dollars_per_petaflop: float = 0.15  # rough cloud GPU blend
    dollars_per_gb_hour: float = 0.002
    moe_comm_overhead: float = 0.05  # fraction of compute cost


@dataclass
class CostEstimate:
    total_params_b: float
    active_params_b: float
    active_flops_per_token: float
    peak_vram_gb: float
    dollars_per_million_tokens: float
    details: Dict[str, float]


def estimate_active_flops_per_token(config: LunaConfig, seq_len: int = 2048) -> float:
    """Approximate active FLOPs per token for one forward pass."""
    d = config.hidden_size
    l_m = config.mamba2_layers
    l_a = config.mla_layers
    e_a = config.num_expert_activated + config.num_shared_experts
    i = config.intermediate_size
    d_inner = d * config.mamba_expand
    d_state = config.mamba_d_state

    # Embed + LM head amortized over seq
    embed = 2 * config.vocab_size * d / max(1, seq_len)

    # Mamba: in/out proj + selective scan (rough)
    mamba = l_m * (
        2 * d * (2 * d_inner)  # in_proj
        + 2 * d_inner * d  # out_proj
        + 2 * d_inner * d_state * 4  # scan-ish
    )

    # MLA: low-rank Q/KV + attention matmuls (rough, seq-dependent attn)
    q_cost = 2 * d * config.q_lora_rank + 2 * config.q_lora_rank * (
        config.n_heads * (config.qk_nope_head_dim + config.qk_rope_head_dim)
    )
    kv_cost = 2 * d * (config.kv_lora_rank + config.n_heads * config.qk_rope_head_dim)
    kv_b = 2 * config.kv_lora_rank * (
        config.n_heads * (config.qk_nope_head_dim + config.v_head_dim)
    )
    attn = 2 * config.n_heads * seq_len * (
        config.qk_nope_head_dim + config.qk_rope_head_dim + config.v_head_dim
    )
    mla = l_a * (q_cost + kv_cost + kv_b + attn)

    # Active MoE SwiGLU: gate + up + down
    moe = (l_m + l_a) * e_a * (6 * d * i)

    # CTM average ticks ≈ max_ticks / 2
    avg_ticks = max(1, config.ctm_max_ticks / 2.0)
    ctm = avg_ticks * (
        2 * d * config.ctm_n_neurons
        + 2 * config.ctm_n_neurons * config.ctm_nlm_hidden
        + 2 * config.ctm_n_neurons * d
    )

    return float(embed + mamba + mla + moe + ctm)


def estimate_peak_vram_gb(
    config: LunaConfig,
    batch_size: int = 1,
    seq_len: int = 2048,
    param_bits: int = 16,
    kv_bits: int = 4,
    activation_bytes_factor: float = 2.0,
) -> float:
    """Rough peak VRAM in GB for training/inference of one replica."""
    total_b, active_b = count_parameters(config)
    # Prefer storing all params (MoE) at param_bits; activations use active path
    params_gb = total_b * 1e9 * (param_bits / 8) / (1024 ** 3)

    d = config.hidden_size
    layers = config.num_hidden_layers
    act_gb = (
        batch_size * seq_len * d * layers * activation_bytes_factor
    ) / (1024 ** 3)

    # Compressed KV for MLA layers
    kv_gb = (
        batch_size
        * seq_len
        * config.mla_layers
        * config.kv_lora_rank
        * (kv_bits / 8)
    ) / (1024 ** 3)

    d_inner = d * config.mamba_expand
    mamba_state_gb = (
        batch_size
        * config.mamba2_layers
        * d_inner
        * config.mamba_d_state
        * 2
    ) / (1024 ** 3)

    return float(params_gb + act_gb + kv_gb + mamba_state_gb)


def estimate_cost(
    config: LunaConfig,
    batch_size: int = 1,
    seq_len: int = 2048,
    param_bits: int = 16,
    kv_bits: Optional[int] = None,
    price: Optional[HardwarePrice] = None,
) -> CostEstimate:
    """Full cost estimate bundle."""
    price = price or HardwarePrice()
    if kv_bits is None:
        kv_bits = 4 if config.use_kv_cache_int4 else 16

    total_b, active_b = count_parameters(config)
    flops = estimate_active_flops_per_token(config, seq_len=seq_len)
    vram = estimate_peak_vram_gb(
        config,
        batch_size=batch_size,
        seq_len=seq_len,
        param_bits=param_bits,
        kv_bits=kv_bits,
    )

    # $/MTok from FLOPs + small MoE comm overhead
    petaflops_per_mtok = (flops * 1e6) / 1e15
    compute_cost = petaflops_per_mtok * price.dollars_per_petaflop
    dollars = compute_cost * (1.0 + price.moe_comm_overhead)

    return CostEstimate(
        total_params_b=total_b,
        active_params_b=active_b,
        active_flops_per_token=flops,
        peak_vram_gb=vram,
        dollars_per_million_tokens=dollars,
        details={
            "param_bits": float(param_bits),
            "kv_bits": float(kv_bits),
            "seq_len": float(seq_len),
            "batch_size": float(batch_size),
        },
    )


def summarize_preset(preset: str, **kwargs) -> CostEstimate:
    config = LunaConfig.from_preset(preset)
    return estimate_cost(config, **kwargs)


if __name__ == "__main__":
    for name in ("tiny", "1b", "7b", "550b"):
        est = summarize_preset(name, seq_len=512 if name == "tiny" else 2048)
        print(
            f"{name:6s} | total={est.total_params_b:8.3f}B "
            f"active={est.active_params_b:8.3f}B "
            f"FLOPs/tok={est.active_flops_per_token:.2e} "
            f"VRAM~{est.peak_vram_gb:.2f}GB "
            f"$/MTok={est.dollars_per_million_tokens:.4f}"
        )
