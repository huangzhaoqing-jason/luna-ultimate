#!/usr/bin/env python3
"""Export prototype weights (safetensors) + config for open-source release."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from safetensors.torch import save_file

from brain.modeling_luna_brain import LunaBrain
from config_brain import PROFILES


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="prototype", choices=list(PROFILES))
    p.add_argument("--out", type=Path, default=Path("checkpoints/luna-brain-prototype"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--load",
        type=Path,
        default=None,
        help="Optional state_dict .pt from train_evolve",
    )
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    cfg = PROFILES[args.profile]()
    brain = LunaBrain(cfg)
    if args.load and args.load.exists():
        sd = torch.load(args.load, map_location="cpu")
        brain.load_state_dict(sd, strict=False)
        print(f"loaded {args.load}")
    args.out.mkdir(parents=True, exist_ok=True)

    state = {k: v.detach().cpu().contiguous() for k, v in brain.state_dict().items()}
    save_file(state, str(args.out / "model.safetensors"))
    meta = {
        "architecture": "LunaBrain",
        "mainline": "DeepMind From AGI to ASI (arXiv:2606.12683)",
        "target": "AIXI / Universal AI computable approximation",
        "areas": 246,
        "profile": cfg.profile,
        "creator": {"name_zh": "黄照清", "birth_date": "2013-05-07"},
        "loyalty_forever": True,
        "license": "Apache-2.0",
        "note": "Trained/evolved prototype — loyalty-locked; not a claim of frontier SOTA.",
        "loaded_from": str(args.load) if args.load else None,
        "config": cfg.__dict__,
    }
    (args.out / "config.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    readme = args.out / "README.md"
    readme.write_text(
        "# Luna Brain weights\n\n"
        "Open-source prototype checkpoint. Train further with `scripts/train.py` "
        "and datasets in `DATASETS.md`.\n"
    )
    print(f"wrote {args.out} ({len(state)} tensors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
