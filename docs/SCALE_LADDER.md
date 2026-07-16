# Scale Ladder & 77B-active Challenge

## Goal

Train up to **550B total** using Apple-style **FlashMoE** as-is (not a custom rewrite).

| | |
|--|--|
| Total weights | ~550B |
| Knife-tip (active) | ~77–80B = Top-K routed + shared experts per token |
| Point | Put the FLOPs/quality on the active tip so same-tier active-budget performance is strongest |

Train the ladder first; at 550b the FlashMoE architecture *is* the knife-tip. Optional CPU/GPU expert prefetch helpers are deploy-only, not a different MoE math.

## Policy

1. Evolve and harden on `tiny` → `1b` → `7b`.
2. Only promote to `550b` / `77b_active` when Pareto archive on `7b` is stable.
3. `evolve.loop` refuses `550b` unless `--allow-large`.
4. Do not replace FlashMoE with ad-hoc streaming “architectures”; keep Top-K + shared + aux/z-loss.

## Commands

```bash
# 1b sprint（真实文本；无 data_path 则随机 token，质量不上）
python scripts/scale_train.py --preset 1b --max_steps 1000 --seq_len 512 \
  --data_path data/corpus.txt --distill

# 直接 train.py
python train.py --preset 1b --modality text --data_path data/corpus.txt \
  --max_steps 1000 --seq_len 512 --output_dir checkpoints/1b_short

# 7b with ZeRO-2
python scripts/scale_train.py --preset 7b --max_steps 5000 \
  --deepspeed configs/ds_config_7b.json --allow-large

# 77B-active challenge (multi-node)
torchrun --nproc_per_node=8 train.py --preset 550b \
  --use_deepspeed --deepspeed_config configs/ds_config_77b.json \
  --batch_size 1 --grad_accum 16 --seq_len 2048
```

中文优化阶梯说明见 `docs/zh-CN/OPTIMIZATION.md`（JEPA 总控 / 评测 collapse+uncertainty / ModelScope）。

## Hardware (honest)

| Preset | What you need |
|--------|----------------|
| `tiny` | CPU / small RAM — smoke + evolve |
| `1b` | ≥24–32GB RAM or 1×24GB GPU (full Adam states) |
| `7b` | multi-GPU ZeRO-2 (`configs/ds_config_7b.json`) |
| `550b` | multi-node ZeRO-3 + EP (`configs/ds_config_77b.json`); **~550B total / ~77–80B active** |

This is not “rewrite FlashMoE to fit 5GB”. FlashMoE stays Apple-style; the knife-tip is **activation sparsity** (Top-K), not a different architecture.

## Acceptance

Cost-normalized proxy (`scripts/run_eval_proxy.py`) plus external suites in `EVALUATION.md`.
Proxy now reports `collapse_rate` + `jepa_uncertainty_mean` + cost.
Under fixed **active**-FLOPs / `$` budget (knife-tip), beat same-budget open SOTA on the weighted suite.
