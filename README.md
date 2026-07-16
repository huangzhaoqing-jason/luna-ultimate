# Luna Evolve

**Hybrid research prototype:** CTM × Mamba2-SSD × MLA × FlashMoE × JEPA  
**Goal:** cost–quality Pareto self-evolution toward GPT-4o-class capability *subsets*.

This is **not** a claim that a single 550B run already beats GPT-4o.  
The default challenge preset is ~550B total / ~77B active; day-to-day work uses `tiny` / `1b` / `7b`.

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [docs/COST_MODEL.md](docs/COST_MODEL.md).

```
Embed → Mamba2 front → MLA back → LM head
         ↑ FlashMoE on every layer
CTM block-inject + CTM-JEPA (future-state prediction)
Outer loop: evolve Arch/Train/Infer/Agent genomes (Pareto)
```

## Presets

| Preset | Use |
|--------|-----|
| `tiny` | Unit tests + evolve smoke |
| `1b` | Single-GPU e2e |
| `7b` | Capability sprint |
| `550b` / `77b_active` | Multi-node challenge ladder |

```bash
python -c "from config import LunaConfig, verify_parameters; verify_parameters(LunaConfig.from_preset('tiny'))"
python cost_model.py
```

## Install

```bash
pip install -e .
# optional training extras
pip install -e ".[train]"
```

## Smoke train (tiny)

```bash
python train.py --preset tiny --smoke --batch_size 2 --seq_len 64 --max_steps 5
```

## Self-evolution loop

```bash
python -m evolve.loop --preset tiny --generations 3 --population 4 --train_steps 5
```

## Scale scripts

```bash
# 1b → 7b ladder helpers
python scripts/scale_train.py --preset 1b --max_steps 100
python scripts/run_eval_proxy.py --preset tiny --checkpoint none
```

## 77B-active challenge

See `configs/ds_config_77b.json` and `scripts/scale_train.py --preset 550b`  
(only after evolve archive shows stable genomes on smaller ladders).

## License

Apache 2.0
