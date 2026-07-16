#!/usr/bin/env python3
"""Proxy evaluation for Luna Evolve (CE + ticks + cost), no external datasets required.

Usage:
  python scripts/run_eval_proxy.py --preset tiny
  python scripts/run_eval_proxy.py --preset tiny --checkpoint ./checkpoints/smoke_final.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import LunaConfig
from cost_model import estimate_cost
from modeling_luna_ultimate import LunaUltimateFused


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", type=str, default="tiny")
    p.add_argument("--checkpoint", type=str, default="none")
    p.add_argument("--seq_len", type=int, default=64)
    p.add_argument("--batches", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--output", type=str, default="./eval_proxy.json")
    args = p.parse_args()

    device = torch.device("cpu")
    config = LunaConfig.from_preset(args.preset)
    model = LunaUltimateFused(config).to(device)

    if args.checkpoint not in ("none", "", "None"):
        ckpt = torch.load(args.checkpoint, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint {args.checkpoint}")

    model.eval()
    total_ce = 0.0
    total_ticks = 0.0
    t0 = time.time()

    with torch.no_grad():
        for _ in range(args.batches):
            ids = torch.randint(0, config.vocab_size, (args.batch_size, args.seq_len))
            labels = ids.clone()
            labels[:, :-1] = ids[:, 1:]
            labels[:, -1] = -100
            out = model(ids, return_all_losses=True, use_ctm_adaptive=True)
            loss, stats = model.compute_loss(out, labels)
            total_ce += stats.get("lm_loss_raw", float(loss))
            total_ticks += stats.get("avg_ticks", float(model.last_avg_ticks))

    elapsed = time.time() - t0
    import math
    avg_ce = total_ce / args.batches
    avg_ticks = total_ticks / args.batches
    cost = estimate_cost(config, batch_size=args.batch_size, seq_len=args.seq_len)
    baseline = math.log(max(2, config.vocab_size))
    quality = 1.0 / (1.0 + math.exp(avg_ce - baseline))

    report = {
        "preset": args.preset,
        "avg_ce": avg_ce,
        "avg_ticks": avg_ticks,
        "quality_proxy": quality,
        "latency_ms_per_batch": (elapsed / args.batches) * 1000,
        "active_flops_per_token": cost.active_flops_per_token,
        "peak_vram_gb": cost.peak_vram_gb,
        "dollars_per_million_tokens": cost.dollars_per_million_tokens,
        "total_params_b": cost.total_params_b,
        "active_params_b": cost.active_params_b,
    }
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
