# Luna Evolve Architecture

## Premise

Luna Evolve is a **research prototype** with a fixed hybrid topology and a searchable genome.
It does **not** claim a one-shot 550B training run that beats GPT-4o.
Success is defined as a **cost–quality Pareto** system that climbs capability ladders
(`tiny` → `1b_active` → `7b_active` → `77b_active`) under a fixed dollar / active-FLOPs budget.

## Inner loop (forward topology)

```
Text (+ optional V-JEPA) → Embed
  → Mamba2-SSD front (context encode, O(1) state)
  → MLA back (deep reason, compressed KV)
  → LM head
Global CTM: adaptive ticks, residual injected every N layers (block inject)
FlashMoE: routed Top-K + shared experts on every layer
CTM-JEPA: predict future neuron state vs stop-grad EMA target of real future
```

| Component | Role | Knobs (ArchGenome) |
|-----------|------|--------------------|
| Mamba2-SSD front | Long-context encode, no KV growth | `mamba_ratio`, `mamba_d_state` |
| MLA back | Deep attention + YaRN | `kv_lora_rank`, `q_lora_rank` |
| FlashMoE | Sparse capacity | `num_routed_experts`, `top_k`, `intermediate_size` |
| CTM | Adaptive internal compute | `ctm_n_neurons`, `ctm_max_ticks`, `ctm_inject_every` |
| CTM-JEPA | Internal consistency | `ctm_jepa_horizon`, `ctm_jepa_ema_decay` |
| V-JEPA | Multimodal (stage 3) | enabled/disabled by stage |

## Outer loop (self-evolution)

AlphaEvolve evolves programs. Luna Evolve evolves **four genomes**:

1. **ArchGenome** — layer split, expert count, CTM ticks, ranks
2. **TrainGenome** — LR group multipliers, Archer KL, aux/z-loss, stage weights
3. **InferGenome** — early-exit thresholds, layer-skip, draft length, quant bits
4. **AgentGenome** — eval weights, self-play mix, distill teacher choice

Fitness (multi-objective, maximize quality, minimize cost):

```
maximize  quality(proxy_eval)
minimize  active_FLOPs_per_token, peak_VRAM_GB, latency_ms
```

Pareto archive keeps non-dominated individuals; champions promote to the next generation baseline.

## Training stages

| Stage | Focus | Frozen |
|-------|-------|--------|
| 1 | Text LM pretrain | V-JEPA |
| 2 | CTM-JEPA + Archer reasoning | V-JEPA |
| 3 | Multimodal alignment | none |

## Acceptance (cost-normalized)

Under a fixed `$` or active-FLOPs budget, weighted score on
GSM8K / MATH / HumanEval / MMLU-STEM / BBH meets or exceeds
a published GPT-4o report (or same-budget open SOTA).

## Key modules

- Model: `modeling_luna_ultimate.py` (`LunaUltimateFused`)
- Config presets: `config.py` (`LunaConfig.from_preset`)
- Cost: `cost_model.py`
- Evolution: `evolve/`
- Train: `train.py`
