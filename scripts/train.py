#!/usr/bin/env python3
"""Minimal open training loop: AIXI hypothesis heads + capability language aux.

Uses synthetic batches by default (smoke). Wire real FineWeb/Dolma via --streaming
when `datasets` is installed and network/cluster available.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from brain.modeling_luna_brain import LunaBrain
from config_brain import PROFILES


def synthetic_batch(cfg, batch_size: int = 4):
    state = torch.randn(batch_size, cfg.d_model)
    goals = torch.randn(2, cfg.d_model)
    # Fake "better action" target for smoke supervision
    target_idx = torch.randint(0, cfg.n_actions, (batch_size,))
    return state, goals, target_idx


def try_load_text_stream(recipe_path: Path, max_samples: int):
    """Optional: stream a few FineWeb samples if datasets is available."""
    try:
        import yaml
        from datasets import load_dataset
    except ImportError:
        return None
    if not recipe_path.exists():
        return None
    recipe = yaml.safe_load(recipe_path.read_text())
    stage = recipe["stages"][0]
    ds_id = stage["datasets"][0]["id"]
    subset = stage["datasets"][0].get("subset")
    try:
        kwargs = {"split": "train", "streaming": True}
        if subset:
            kwargs["name"] = subset
        ds = load_dataset(ds_id, **kwargs)
    except Exception as e:
        print(f"[data] skip stream {ds_id}: {e}")
        return None
    texts = []
    for i, row in enumerate(ds):
        texts.append(row.get("text") or row.get("content") or "")
        if i + 1 >= max_samples:
            break
    return texts


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="prototype", choices=list(PROFILES))
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--data-config", type=Path, default=ROOT / "configs/data_recipe.yaml")
    p.add_argument("--out", type=Path, default=Path("checkpoints/luna-brain-train"))
    args = p.parse_args(argv)

    cfg = PROFILES[args.profile]()
    brain = LunaBrain(cfg)
    opt = torch.optim.AdamW(brain.parameters(), lr=cfg.learning_rate)

    texts = try_load_text_stream(args.data_config, max_samples=8)
    if texts:
        print(f"[data] streamed {len(texts)} samples from recipe")
    else:
        print("[data] using synthetic batches (install datasets+pyyaml for FineWeb stream)")

    history = []
    brain.train()
    for step in range(args.steps):
        state, goals, target_idx = synthetic_batch(cfg)
        out = brain(state, goal_states=goals, creator_aligned=True)
        # Supervise AIXI discrete choice toward synthetic target + keep returns finite
        logits = out["expected_returns"]
        # expected_returns is [B]; rebuild soft preference via action indices CE proxy
        # Use capability language head if present
        loss = F.mse_loss(out["actions"], torch.zeros_like(out["actions"]))
        if "logits" in out:
            loss = loss + 0.01 * out["logits"].pow(2).mean()
        # Prefer target action index via one-hot on returns as ranking signal
        # Expand: run aixi returns matrix — re-forward internal
        ret, _ = brain.aixi._predict_returns(out["state"], creator_aligned=True)
        loss = loss + F.cross_entropy(ret, target_idx)

        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 20 == 0 or step == args.steps - 1:
            history.append({"step": step, "loss": float(loss.item())})
            print(f"step {step} loss={loss.item():.4f}")

        if step > 0 and step % 50 == 0:
            brain.evolve_once({"task_score": max(0.0, 1.0 - loss.item()), "memory_gb": 0.2, "memory_budget_gb": 8.0})

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(brain.state_dict(), args.out / "model.pt")
    (args.out / "train_history.json").write_text(json.dumps(history, indent=2))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
