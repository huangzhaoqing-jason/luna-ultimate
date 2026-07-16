#!/usr/bin/env python3
"""Scale-ladder training helper: tiny → 1b → 7b → (optional) 550b.

Usage:
  python scripts/scale_train.py --preset 1b --max_steps 100
  python scripts/scale_train.py --preset 7b --max_steps 500
  python scripts/scale_train.py --preset 550b --allow-large --deepspeed configs/ds_config_77b.json
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LADDER = ("tiny", "1b", "7b", "550b")


def main():
    p = argparse.ArgumentParser(description="Luna scale-ladder trainer")
    p.add_argument("--preset", type=str, default="1b", choices=list(LADDER) + ["77b_active"])
    p.add_argument("--max_steps", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--stage", type=int, default=1)
    p.add_argument("--output_dir", type=str, default="./checkpoints/scale")
    p.add_argument("--deepspeed", type=str, default=None)
    p.add_argument("--allow-large", action="store_true")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    preset = "550b" if args.preset == "77b_active" else args.preset
    if preset == "550b" and not args.allow_large:
        print("Refusing 550b without --allow-large. Stabilize 1b/7b first.", file=sys.stderr)
        sys.exit(2)

    cmd = [
        sys.executable, str(ROOT / "train.py"),
        "--preset", preset,
        "--max_steps", str(args.max_steps),
        "--batch_size", str(args.batch_size),
        "--seq_len", str(args.seq_len),
        "--stage", str(args.stage),
        "--output_dir", f"{args.output_dir}/{preset}",
        "--grad_accum", "1",
        "--log_every", "1",
        "--save_every", str(max(1, args.max_steps)),
    ]
    if args.smoke or preset == "tiny":
        cmd.append("--smoke")
    if args.deepspeed:
        cmd.extend(["--use_deepspeed", "--deepspeed_config", args.deepspeed])

    print("Running:", " ".join(cmd))
    raise SystemExit(subprocess.call(cmd, cwd=str(ROOT)))


if __name__ == "__main__":
    main()
