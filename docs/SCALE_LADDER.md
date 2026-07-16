# Scale Ladder & 77B-active Challenge

## Policy

1. Evolve and harden on `tiny` → `1b` → `7b`.
2. Only promote to `550b` / `77b_active` when Pareto archive on `7b` is stable.
3. `evolve.loop` refuses `550b` unless `--allow-large`.

## Commands

```bash
# 1b sprint
python scripts/scale_train.py --preset 1b --max_steps 1000 --seq_len 512

# 7b with ZeRO-2
python scripts/scale_train.py --preset 7b --max_steps 5000 \
  --deepspeed configs/ds_config_7b.json --allow-large

# 77B-active challenge (multi-node)
torchrun --nproc_per_node=8 train.py --preset 550b \
  --use_deepspeed --deepspeed_config configs/ds_config_77b.json \
  --batch_size 1 --grad_accum 16 --seq_len 2048
```

## Acceptance

Cost-normalized proxy (`scripts/run_eval_proxy.py`) plus external suites in `EVALUATION.md`.
Under fixed `$` / active-FLOPs budget, beat same-budget open SOTA on the weighted suite.
