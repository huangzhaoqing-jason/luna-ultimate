#!/usr/bin/env python3
"""Publish exported checkpoint + this repo pointer to Hugging Face Hub.

Requires: pip install huggingface_hub && HF_TOKEN in env.
ModelScope upload/download scripts were intentionally removed.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=Path, required=True, help="export dir with model.safetensors")
    p.add_argument("--repo", required=True, help="HF repo id, e.g. user/luna-brain")
    p.add_argument("--private", action="store_true")
    args = p.parse_args(argv)

    if not os.environ.get("HF_TOKEN"):
        print("HF_TOKEN not set — cannot upload. Export is local only.", file=sys.stderr)
        return 2
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("pip install huggingface_hub", file=sys.stderr)
        return 2

    if not (args.dir / "model.safetensors").exists():
        print("missing model.safetensors; run scripts/export_weights.py first", file=sys.stderr)
        return 2

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(args.repo, private=args.private, exist_ok=True)
    api.upload_folder(folder_path=str(args.dir), repo_id=args.repo, repo_type="model")
    print(f"uploaded {args.dir} -> https://huggingface.co/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
