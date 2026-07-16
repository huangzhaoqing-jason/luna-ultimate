# Luna Evolve Cost Model

## Formulas

Let `d = hidden_size`, `L_m = mamba2_layers`, `L_a = mla_layers`,
`E_a = top_k + num_shared`, `I = intermediate_size`.

### Active FLOPs / token (approx)

```
FLOPs ≈ 2 * (
  vocab*d/seq_amortized          # embed + lm_head (amortized)
  + L_m * (mamba_proj + scan)    # Mamba front
  + L_a * (mla_qkv + attn)       # MLA back
  + (L_m+L_a) * E_a * (3*d*I)    # active SwiGLU experts
  + ctm_ticks * ctm_cost         # adaptive CTM
)
```

Implemented in `cost_model.estimate_active_flops_per_token`.

### Peak memory (approx)

```
Mem ≈ params_bytes(quant_bits)
    + activation_bytes(batch, seq, d, layers)
    + kv_cache_bytes(seq, kv_lora_rank, mla_layers, kv_bits)
    + mamba_state_bytes(batch, d_inner, d_state, L_m)
```

### $/token

```
$/token = FLOPs * ($/FLOP_hardware) + moe_ep_comm_overhead
```

Default hardware unit prices are placeholders in `cost_model.HardwarePrice`.

## Scale ladder

| Preset | Rough active params | Typical hardware | Role |
|--------|---------------------|------------------|------|
| `tiny` | ~1–5M | CPU / 1×8–24GB | unit tests, evolve smoke |
| `1b` | ~0.5–1.5B active | 1×24GB | e2e train/infer |
| `7b` | ~5–10B active | 1–8×80GB | capability subset sprint |
| `550b` / `77b_active` | ~70–85B active | multi-node EP/PP | 4o-subset challenge |

Evolution searches **only** on ladder presets (and genomes within preset bounds).
Instantiating full 550B before Phase-2 genome stability is disallowed by policy in `evolve/loop.py`.

## Pareto axes

1. `quality` — proxy CE inverse or eval score in `[0,1]`
2. `active_flops` — lower better
3. `peak_vram_gb` — lower better
4. `latency_ms` — lower better (optional in smoke)

Champion selection: max `quality - λ·norm(flops) - μ·norm(vram) - ν·norm(latency)`.
