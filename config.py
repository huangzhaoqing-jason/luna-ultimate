"""Luna-Ultimate / Luna Evolve configuration with scale presets."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Optional, Tuple


PRESET_NAMES = ("tiny", "1b", "7b", "550b", "77b_active")


@dataclass
class LunaConfig:
    """Configuration for Luna hybrid models.

    Default values match the 550B / ~77B-active challenge preset.
    Use ``LunaConfig.from_preset(...)`` for ladder scales.
    """

    # ==================== Vocabulary & Embedding ====================
    vocab_size: int = 151936
    hidden_size: int = 8192
    pad_token_id: int = 0

    # ==================== Architecture ====================
    num_hidden_layers: int = 32
    mamba2_layers: int = 12
    mla_layers: int = 20

    # ==================== Mamba2-SSD ====================
    mamba_d_state: int = 128
    mamba_d_conv: int = 4
    mamba_expand: int = 2

    # ==================== MLA ====================
    n_heads: int = 32
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    kv_lora_rank: int = 1024
    q_lora_rank: int = 3072

    # ==================== CTM ====================
    ctm_n_neurons: int = 4096
    ctm_nlm_hidden: int = 128
    ctm_max_ticks: int = 4
    ctm_entropy_thresholds: Tuple[float, float, float] = (0.3, 0.6, 0.9)
    ctm_inject_every: int = 4  # block-level CTM residual reuse

    # ==================== CTM-JEPA ====================
    ctm_jepa_enabled: bool = True
    ctm_jepa_horizon: int = 1
    ctm_jepa_ema_decay: float = 0.996

    # ==================== V-JEPA ====================
    vjepa_enabled: bool = True
    vjepa_config: dict = field(default_factory=lambda: {
        "img_size": (224, 224),
        "patch_size": (2, 16, 16),
        "in_channels": 3,
        "embed_dim": 1024,
        "encoder_depth": 24,
        "predictor_depth": 6,
        "num_heads": 16,
        "mask_ratio": 0.75,
        "use_target_encoder": True,
        "ema_decay": 0.996,
    })

    # ==================== FlashMoE ====================
    num_routed_experts: int = 48
    num_shared_experts: int = 2
    num_expert_activated: int = 4
    intermediate_size: int = 13568
    moe_capacity_factor: float = 1.25
    moe_aux_loss_coeff: float = 0.01
    moe_z_loss_coeff: float = 0.001

    # ==================== RoPE / YaRN ====================
    rope_theta: float = 10000.0
    yarn_factor: float = 8.0
    max_position_embeddings: int = 131072

    # ==================== Normalization ====================
    rms_norm_eps: float = 1e-6

    # ==================== Archer ====================
    archer_knowledge_kl_weight: float = 0.1
    archer_reasoning_kl_weight: float = 0.001
    archer_reasoning_clip_threshold: float = 0.3
    archer_entropy_threshold: float = 0.5

    # ==================== Training ====================
    learning_rate: float = 1e-4
    warmup_steps: int = 2000
    max_steps: int = 100000
    global_batch_size: int = 4_000_000
    gradient_accumulation_steps: int = 8
    mixed_precision: str = "bf16"
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    max_grad_norm: float = 1.0
    preset_name: str = "550b"

    # ==================== Inference ====================
    use_speculative_decoding: bool = True
    num_draft_tokens: int = 4
    use_kv_cache_int4: bool = True
    use_ctm_adaptive_early_exit: bool = True
    use_dynamic_layer_skip: bool = True
    layer_skip_prob_threshold: float = 0.7

    def __post_init__(self):
        assert self.mamba2_layers + self.mla_layers == self.num_hidden_layers, (
            f"Layers mismatch: {self.mamba2_layers}+{self.mla_layers}"
            f"!={self.num_hidden_layers}"
        )
        assert self.hidden_size % self.n_heads == 0, (
            f"hidden_size {self.hidden_size} must be divisible by n_heads {self.n_heads}"
        )
        assert self.ctm_inject_every >= 1
        assert self.num_expert_activated <= self.num_routed_experts

    @classmethod
    def from_preset(cls, name: str, **overrides) -> "LunaConfig":
        """Build a config from a named scale ladder preset."""
        key = name.lower().replace("luna_", "").replace("-", "_")
        if key == "77b_active":
            key = "550b"
        if key not in _PRESETS:
            raise ValueError(
                f"Unknown preset {name!r}. Choose from: {', '.join(PRESET_NAMES)}"
            )
        base = cls(**_PRESETS[key])
        if overrides:
            base = replace(base, **overrides)
            base.__post_init__()
        return base

    def to_dict(self) -> Dict:
        return {f.name: getattr(self, f.name) for f in self.__dataclass_fields__.values()}


# Scale ladder presets (genomes mutate within these bounds)
_PRESETS: Dict[str, dict] = {
    "tiny": {
        "preset_name": "tiny",
        "vocab_size": 4096,
        "hidden_size": 256,
        "num_hidden_layers": 4,
        "mamba2_layers": 2,
        "mla_layers": 2,
        "mamba_d_state": 16,
        "mamba_d_conv": 4,
        "mamba_expand": 2,
        "n_heads": 4,
        "qk_nope_head_dim": 32,
        "qk_rope_head_dim": 16,
        "v_head_dim": 32,
        "kv_lora_rank": 64,
        "q_lora_rank": 128,
        "ctm_n_neurons": 64,
        "ctm_nlm_hidden": 32,
        "ctm_max_ticks": 2,
        "ctm_entropy_thresholds": (0.2, 0.5, 0.8),
        "ctm_inject_every": 2,
        "ctm_jepa_enabled": True,
        "vjepa_enabled": False,
        "vjepa_config": {
            "img_size": (32, 32),
            "patch_size": (1, 8, 8),
            "in_channels": 3,
            "embed_dim": 64,
            "encoder_depth": 2,
            "predictor_depth": 1,
            "num_heads": 2,
            "mask_ratio": 0.5,
            "use_target_encoder": True,
            "ema_decay": 0.99,
        },
        "num_routed_experts": 4,
        "num_shared_experts": 1,
        "num_expert_activated": 2,
        "intermediate_size": 512,
        "max_position_embeddings": 2048,
        "learning_rate": 3e-4,
        "warmup_steps": 20,
        "max_steps": 200,
        "use_speculative_decoding": False,
        "use_dynamic_layer_skip": False,
    },
    "1b": {
        "preset_name": "1b",
        "vocab_size": 32000,
        "hidden_size": 1536,
        "num_hidden_layers": 16,
        "mamba2_layers": 8,
        "mla_layers": 8,
        "mamba_d_state": 64,
        "mamba_expand": 2,
        "n_heads": 12,
        "qk_nope_head_dim": 64,
        "qk_rope_head_dim": 32,
        "v_head_dim": 64,
        "kv_lora_rank": 256,
        "q_lora_rank": 768,
        "ctm_n_neurons": 512,
        "ctm_nlm_hidden": 64,
        "ctm_max_ticks": 3,
        "ctm_inject_every": 4,
        "vjepa_enabled": False,
        "vjepa_config": {
            "img_size": (128, 128),
            "patch_size": (2, 16, 16),
            "in_channels": 3,
            "embed_dim": 384,
            "encoder_depth": 6,
            "predictor_depth": 2,
            "num_heads": 6,
            "mask_ratio": 0.75,
            "use_target_encoder": True,
            "ema_decay": 0.996,
        },
        "num_routed_experts": 8,
        "num_shared_experts": 1,
        "num_expert_activated": 2,
        "intermediate_size": 4096,
        "max_position_embeddings": 8192,
        "learning_rate": 2e-4,
        "warmup_steps": 200,
        "max_steps": 10000,
    },
    "7b": {
        "preset_name": "7b",
        "vocab_size": 65536,
        "hidden_size": 4096,
        "num_hidden_layers": 28,
        "mamba2_layers": 12,
        "mla_layers": 16,
        "mamba_d_state": 128,
        "mamba_expand": 2,
        "n_heads": 32,
        "qk_nope_head_dim": 96,
        "qk_rope_head_dim": 32,
        "v_head_dim": 96,
        "kv_lora_rank": 512,
        "q_lora_rank": 1536,
        "ctm_n_neurons": 2048,
        "ctm_nlm_hidden": 128,
        "ctm_max_ticks": 4,
        "ctm_inject_every": 4,
        "vjepa_enabled": False,
        "vjepa_config": {
            "img_size": (224, 224),
            "patch_size": (2, 16, 16),
            "in_channels": 3,
            "embed_dim": 768,
            "encoder_depth": 12,
            "predictor_depth": 4,
            "num_heads": 12,
            "mask_ratio": 0.75,
            "use_target_encoder": True,
            "ema_decay": 0.996,
        },
        "num_routed_experts": 16,
        "num_shared_experts": 2,
        "num_expert_activated": 2,
        "intermediate_size": 11008,
        "max_position_embeddings": 32768,
        "learning_rate": 1.5e-4,
        "warmup_steps": 500,
        "max_steps": 50000,
    },
    "550b": {
        "preset_name": "550b",
        # defaults on LunaConfig already match 550B / ~77B active
    },
}


def count_parameters(config: LunaConfig) -> Tuple[float, float]:
    """Count total and active parameters (billions)."""
    d = config.hidden_size
    n_layers = config.num_hidden_layers
    n_mamba = config.mamba2_layers
    n_mla = config.mla_layers

    embed = config.vocab_size * d

    d_inner = d * config.mamba_expand
    d_inner_double = d_inner * 2
    mamba_in_proj = d * d_inner_double * n_mamba
    mamba_out_proj = d_inner * d * n_mamba
    mamba_conv = config.mamba_d_conv * d_inner * n_mamba
    mamba_x_proj = (d_inner + config.mamba_d_state * 2) * d_inner * n_mamba
    mamba_dt_proj = d_inner * d_inner * n_mamba
    mamba_A = d_inner * config.mamba_d_state * n_mamba
    mamba_D = d_inner * n_mamba
    mamba_total = (
        mamba_in_proj + mamba_out_proj + mamba_conv
        + mamba_x_proj + mamba_dt_proj + mamba_A + mamba_D
    )

    mla_qa = d * config.q_lora_rank * n_mla
    mla_qb = config.q_lora_rank * (
        config.n_heads * (config.qk_nope_head_dim + config.qk_rope_head_dim)
    ) * n_mla
    kv_a_out = config.kv_lora_rank + config.n_heads * config.qk_rope_head_dim
    mla_kva = d * kv_a_out * n_mla
    mla_kvb = config.kv_lora_rank * (
        config.n_heads * (config.qk_nope_head_dim + config.v_head_dim)
    ) * n_mla
    mla_o = (config.n_heads * config.v_head_dim) * d * n_mla
    mla_total = mla_qa + mla_qb + mla_kva + mla_kvb + mla_o

    ctm_synapse = d * config.ctm_n_neurons
    ctm_nlm = config.ctm_n_neurons * config.ctm_nlm_hidden * 2
    ctm_proj = config.ctm_n_neurons * d
    ctm_total = ctm_synapse + ctm_nlm + ctm_proj

    n_experts = config.num_routed_experts + config.num_shared_experts
    per_expert = d * config.intermediate_size * 2 + config.intermediate_size * d
    per_router = d * config.num_routed_experts
    moe_total = (per_expert * n_experts + per_router) * n_layers

    norm_total = n_layers * 2 * d
    final_norm = d
    lm_head = d * config.vocab_size

    total = (
        embed + mamba_total + mla_total + ctm_total
        + moe_total + norm_total + final_norm + lm_head
    )

    active_experts = config.num_expert_activated + config.num_shared_experts
    active_m = (per_expert * active_experts + per_router) * n_layers
    active_total = (
        active_m + mamba_total + mla_total + ctm_total
        + embed + norm_total + final_norm + lm_head
    )

    return total / 1e9, active_total / 1e9


def verify_parameters(config: Optional[LunaConfig] = None) -> None:
    """Print parameter verification for the given (or default) config."""
    if config is None:
        config = LunaConfig()
    total_b, active_b = count_parameters(config)
    print(f"{'=' * 60}")
    print("  Luna Parameter Verification")
    print(f"  Preset:              {config.preset_name}")
    print(f"  Total Parameters:    {total_b:.4f}B")
    print(f"  Active Parameters:   {active_b:.4f}B")
    print(f"  Activation Ratio:    {active_b / max(total_b, 1e-12) * 100:.1f}%")
    print(f"{'=' * 60}")
    if config.preset_name in ("550b", "77b_active"):
        if 545.0 <= total_b <= 560.0:
            print(f"  [PASS] Total params {total_b:.2f}B within [545B, 560B]")
        else:
            print(f"  [WARN] Total params {total_b:.2f}B outside [545B, 560B]")
        if 70.0 <= active_b <= 85.0:
            print(f"  [PASS] Active params {active_b:.2f}B within [70B, 85B]")
        else:
            print(f"  [WARN] Active params {active_b:.2f}B outside [70B, 85B]")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    for name in ("tiny", "1b", "7b", "550b"):
        verify_parameters(LunaConfig.from_preset(name))
