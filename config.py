"""Luna-Ultimate Configuration Module.

Contains LunaConfig dataclass with all hyperparameters and a parameter counting
utility to verify the model stays within the 545B-560B range.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple
import math


@dataclass
class LunaConfig:
    """Ultimate configuration for Luna-Ultimate 550B MoE model.

    All hyperparameters are verified against the architecture specification.
    The count_parameters() function validates total params in [545B, 560B].
    """

    # ==================== Vocabulary & Embedding ====================
    vocab_size: int = 151936
    hidden_size: int = 8192
    pad_token_id: int = 0

    # ==================== Architecture ====================
    num_hidden_layers: int = 32
    mamba2_layers: int = 12          # Layers 1-12 Mamba2-SSD
    mla_layers: int = 20             # Layers 13-32 MLA

    # ==================== Mamba2-SSD ====================
    mamba_d_state: int = 128
    mamba_d_conv: int = 4
    mamba_expand: int = 2

    # ==================== MLA (Multi-head Latent Attention) ====================
    n_heads: int = 32
    qk_nope_head_dim: int = 128       # Non-RoPE key dimension per head
    qk_rope_head_dim: int = 64        # RoPE key dimension per head
    v_head_dim: int = 128            # Value dimension per head
    kv_lora_rank: int = 1024         # KV latent compression dimension
    q_lora_rank: int = 3072          # Q latent compression dimension

    # ==================== CTM (Continuous Thought Module) ====================
    ctm_n_neurons: int = 4096
    ctm_nlm_hidden: int = 128        # NLM internal hidden
    ctm_max_ticks: int = 4
    ctm_entropy_thresholds: Tuple[float, float, float] = (0.3, 0.6, 0.9)

    # ==================== FlashMoE ====================
    num_routed_experts: int = 48     # Adjusted to hit 550B target
    num_shared_experts: int = 2
    num_expert_activated: int = 4    # Top-K
    intermediate_size: int = 13568   # Adjusted to hit 550B target
    moe_capacity_factor: float = 1.25
    moe_aux_loss_coeff: float = 0.01

    # ==================== RoPE / YaRN ====================
    rope_theta: float = 10000.0
    yarn_factor: float = 8.0
    max_position_embeddings: int = 131072  # 128K

    # ==================== Normalization ====================
    rms_norm_eps: float = 1e-6

    # ==================== Archer Entropy-Aware Training ====================
    archer_knowledge_kl_weight: float = 0.1
    archer_reasoning_kl_weight: float = 0.001
    archer_reasoning_clip_threshold: float = 0.3
    archer_entropy_threshold: float = 0.5  # Above this = reasoning

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

    # ==================== Inference ====================
    use_speculative_decoding: bool = True
    num_draft_tokens: int = 4
    use_kv_cache_int4: bool = True
    use_ctm_adaptive_early_exit: bool = True
    use_dynamic_layer_skip: bool = True
    layer_skip_prob_threshold: float = 0.7

    def __post_init__(self):
        """Validate configuration consistency."""
        assert self.mamba2_layers + self.mla_layers == self.num_hidden_layers, \
            f"Layers mismatch: {self.mamba2_layers}+{self.mla_layers}!={self.num_hidden_layers}"
        assert self.hidden_size % self.n_heads == 0, \
            f"hidden_size {self.hidden_size} must be divisible by n_heads {self.n_heads}"


def count_parameters(config: LunaConfig) -> Tuple[float, float]:
    """Count total and active parameters for the Luna-Ultimate model.

    Returns:
        (total_params_billions, active_params_billions)
    """
    d = config.hidden_size                # 8192
    n_layers = config.num_hidden_layers   # 32
    n_mamba = config.mamba2_layers        # 12
    n_mla = config.mla_layers             # 20

    # 1. Embedding
    embed = config.vocab_size * d  # 151936 * 8192

    # 2. Mamba2 layers (1-12)
    d_inner = d * config.mamba_expand        # 8192 * 2 = 16384
    d_inner_double = d_inner * 2             # 32768 (in_proj output)
    mamba_in_proj = d * d_inner_double * n_mamba
    mamba_out_proj = d_inner * d * n_mamba
    mamba_conv = config.mamba_d_conv * d_inner * n_mamba
    mamba_x_proj = (d_inner + config.mamba_d_state * 2) * d_inner * n_mamba
    mamba_dt_proj = d_inner * d_inner * n_mamba
    mamba_A = d_inner * config.mamba_d_state * n_mamba
    mamba_D = d_inner * n_mamba

    mamba_total = (mamba_in_proj + mamba_out_proj + mamba_conv +
                   mamba_x_proj + mamba_dt_proj + mamba_A + mamba_D)

    # 3. MLA layers (13-32)
    mla_qa = d * config.q_lora_rank * n_mla
    mla_qb = config.q_lora_rank * (config.n_heads * (config.qk_nope_head_dim + config.qk_rope_head_dim)) * n_mla
    kv_a_out = config.kv_lora_rank + config.n_heads * config.qk_rope_head_dim
    mla_kva = d * kv_a_out * n_mla
    mla_kvb = config.kv_lora_rank * (config.n_heads * (config.qk_nope_head_dim + config.v_head_dim)) * n_mla
    mla_o = (config.n_heads * config.v_head_dim) * d * n_mla

    mla_total = mla_qa + mla_qb + mla_kva + mla_kvb + mla_o

    # 4. CTM (global)
    ctm_synapse = d * config.ctm_n_neurons
    ctm_nlm = config.ctm_n_neurons * config.ctm_nlm_hidden * 2
    ctm_proj = config.ctm_n_neurons * d
    ctm_total = ctm_synapse + ctm_nlm + ctm_proj

    # 5. FlashMoE (32 layers)
    n_experts = config.num_routed_experts + config.num_shared_experts
    per_expert = d * config.intermediate_size * 2 + config.intermediate_size * d
    per_router = d * n_experts
    moe_total = (per_expert * n_experts + per_router) * n_layers

    # 6. RMS Norm layers
    norm_total = n_layers * 2 * d

    # 7. Final norm + lm_head
    final_norm = d
    lm_head = d * config.vocab_size

    total = (embed + mamba_total + mla_total + ctm_total +
             moe_total + norm_total + final_norm + lm_head)

    # Active parameters: 6 experts/layer + attention + CTM + embedding
    active_experts = config.num_expert_activated + config.num_shared_experts
    active_m = (per_expert * active_experts + per_router) * n_layers
    active_total = (active_m + mamba_total + mla_total + ctm_total +
                    embed + norm_total + final_norm + lm_head)

    total_b = total / 1e9
    active_b = active_total / 1e9

    return total_b, active_b


def verify_parameters(config: Optional[LunaConfig] = None) -> None:
    """Verify that the model parameters fall within the target [545B, 560B] range."""
    if config is None:
        config = LunaConfig()
    total_b, active_b = count_parameters(config)
    print(f"{'='*60}")
    print(f"  Luna-Ultimate Parameter Verification")
    print(f"{'='*60}")
    print(f"  Total Parameters:    {total_b:.2f}B")
    print(f"  Active Parameters:   {active_b:.2f}B")
    print(f"  Activation Ratio:    {active_b/total_b*100:.1f}%")
    print(f"{'='*60}")
    if 545.0 <= total_b <= 560.0:
        print(f"  [PASS] Total params {total_b:.2f}B within target [545B, 560B]")
    else:
        print(f"  [WARN] Total params {total_b:.2f}B outside target [545B, 560B]")
    if 70.0 <= active_b <= 85.0:
        print(f"  [PASS] Active params {active_b:.2f}B within target [70B, 85B]")
    else:
        print(f"  [WARN] Active params {active_b:.2f}B outside target [70B, 85B]")
    print(f"{'='*60}")


if __name__ == "__main__":
    verify_parameters()