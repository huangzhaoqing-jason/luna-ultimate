#!/usr/bin/env python3
"""Proxy evaluation for Luna Evolve: CE + ticks + cost + collapse + JEPA uncertainty.

Usage:
  python scripts/run_eval_proxy.py --preset tiny
  python scripts/run_eval_proxy.py --preset tiny --checkpoint ./checkpoints/smoke_final.pt
"""

from __future__ import annotations

import argparse
import json
import math
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

# 代理子集提示（GSM8K / HumanEval / MMLU 风格，无外部数据集）
_PROXY_PROMPTS = [
    "what is 2+2",  # simple
    "prove the theorem by induction on n",  # math/reasoning → higher ticks
    "write a python function to sort a list",  # code
    "what is the capital of France",  # knowledge
    "git commit and push the changes",  # action
]


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
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint {args.checkpoint}")

    model.eval()
    total_ce = 0.0
    total_ticks = 0.0
    uncerts: list[float] = []
    collapses = 0
    gen_n = 0
    t0 = time.time()

    with torch.no_grad():
        for i in range(args.batches):
            ids = torch.randint(0, config.vocab_size, (args.batch_size, args.seq_len))
            labels = ids.clone()
            labels[:, :-1] = ids[:, 1:]
            labels[:, -1] = -100
            route = _PROXY_PROMPTS[i % len(_PROXY_PROMPTS)]
            out = model(
                ids,
                return_all_losses=True,
                use_ctm_adaptive=True,
                route_text=route,
            )
            loss, stats = model.compute_loss(out, labels)
            total_ce += stats.get("lm_loss_raw", float(loss))
            total_ticks += stats.get("avg_ticks", float(model.last_avg_ticks))
            if out.get("jepa_uncertainty") is not None:
                uncerts.append(float(out["jepa_uncertainty"]))

            # collapse 代理：短 generate_meaning_first
            prompt_ids = ids[:1, :16]
            gen = model.generate_meaning_first(
                prompt_ids, max_new_tokens=16, route_text=route
            )
            report = gen["collapse"]
            gen_n += 1
            if getattr(report, "collapsed", False):
                collapses += 1

    elapsed = time.time() - t0
    avg_ce = total_ce / args.batches
    avg_ticks = total_ticks / args.batches
    cost = estimate_cost(config, batch_size=args.batch_size, seq_len=args.seq_len)
    baseline = math.log(max(2, config.vocab_size))
    quality = 1.0 / (1.0 + math.exp(avg_ce - baseline))
    collapse_rate = collapses / max(1, gen_n)
    avg_uncert = sum(uncerts) / max(1, len(uncerts)) if uncerts else 0.0

    # evolve 晋升门：质量↑ 且 collapse 不过高（安全另见 safety suite = 1.0）
    promote_ok = quality > 0.15 and collapse_rate < 0.95

    report = {
        "preset": args.preset,
        "avg_ce": avg_ce,
        "avg_ticks": avg_ticks,
        "quality_proxy": quality,
        "collapse_rate": collapse_rate,
        "jepa_uncertainty_mean": avg_uncert,
        "promote_ok_proxy": promote_ok,
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
