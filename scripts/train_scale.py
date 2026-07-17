#!/usr/bin/env python3
"""Scaled open-data training entry (FineWeb sample stream when available).

Uses synthetic fallback if datasets/network unavailable. Loyalty locked.
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
from brain.safety.loyalty import assert_loyalty_intact, loyalty_audit_line, loyalty_preference_loss
from config_brain import PROFILES


def stream_fineweb(max_samples: int = 64):
    try:
        from datasets import load_dataset
    except ImportError:
        return None
    try:
        ds = load_dataset(
            "HuggingFaceFW/fineweb",
            name="sample-10BT",
            split="train",
            streaming=True,
        )
    except Exception as e:
        print("[data]", e)
        return None
    texts = []
    for i, row in enumerate(ds):
        texts.append((row.get("text") or "")[:512])
        if i + 1 >= max_samples:
            break
    return texts


def text_to_state(text: str, d: int) -> torch.Tensor:
    """Deterministic bag-of-bytes embedding (no external tokenizer required)."""
    v = torch.zeros(d)
    b = text.encode("utf-8", errors="ignore")[:d]
    for i, ch in enumerate(b):
        v[i % d] += (ch / 255.0)
    if v.norm() > 0:
        v = v / v.norm()
    return v


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="prototype", choices=list(PROFILES))
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--out", type=Path, default=Path("checkpoints/luna-brain-scaled"))
    args = p.parse_args(argv)

    assert_loyalty_intact()
    cfg = PROFILES[args.profile]()
    brain = LunaBrain(cfg)
    opt = torch.optim.AdamW(brain.parameters(), lr=cfg.learning_rate)

    texts = stream_fineweb(args.samples)
    if texts:
        print(f"[data] FineWeb stream n={len(texts)}")
        source = "fineweb_sample"
    else:
        texts = [f"synthetic document {i} loyalty 黄照清" for i in range(args.samples)]
        source = "synthetic"
        print("[data] synthetic fallback")

    hist = []
    brain.train()
    for step in range(args.steps):
        assert_loyalty_intact()
        batch_txt = [texts[step % len(texts)], texts[(step + 1) % len(texts)]]
        state = torch.stack([text_to_state(t, cfg.d_model) for t in batch_txt], dim=0)
        out = brain(state, creator_aligned=True)
        ret, _ = brain.aixi._predict_returns(out["state"], creator_aligned=True)
        target = torch.zeros(state.shape[0], dtype=torch.long)
        loss = F.cross_entropy(ret, target)
        loss = loss + 0.5 * loyalty_preference_loss(
            ret, target, torch.ones(state.shape[0])
        )
        # SiFu energy finite
        prompt = torch.randint(0, min(32, cfg.speech_vocab_size), (1, 4))
        sout = brain(state[:1], prompt_ids=prompt, creator_aligned=True)
        if "speech_energies" in sout:
            loss = loss + 0.01 * (-sout["speech_energies"].log_softmax(-1).max(dim=-1).values.mean())

        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 20 == 0 or step == args.steps - 1:
            row = {"step": step, "loss": float(loss.item()), "source": source}
            hist.append(row)
            print(row)

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(brain.state_dict(), args.out / "model.pt")
    (args.out / "history.json").write_text(json.dumps(hist, indent=2))
    (args.out / "LOYALTY.txt").write_text(loyalty_audit_line() + "\n")
    print("wrote", args.out, loyalty_audit_line())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
