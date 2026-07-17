#!/usr/bin/env python3
"""Create/update a ModelScope model repo and upload exported weights.

Requires MODELSCOPE_API_TOKEN. Default owner namespace matches the token account.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=Path, required=True)
    p.add_argument(
        "--repo",
        default="huang18928827157/luna-brain",
        help="ModelScope model id owner/name",
    )
    p.add_argument(
        "--delete-old",
        default="",
        help="Optional old model id to delete before upload",
    )
    p.add_argument("--chinese-name", default="Luna类脑AIXI原型")
    args = p.parse_args(argv)

    token = os.environ.get("MODELSCOPE_API_TOKEN")
    if not token:
        print("MODELSCOPE_API_TOKEN not set", file=sys.stderr)
        return 2
    if not (args.dir / "model.safetensors").exists():
        print("missing model.safetensors; run scripts/export_weights.py", file=sys.stderr)
        return 2

    from modelscope.hub.api import HubApi

    api = HubApi()
    api.login(token)

    if args.delete_old:
        try:
            api.delete_model(args.delete_old)
            print(f"deleted {args.delete_old}")
        except Exception as e:
            print(f"delete_old skip/fail {args.delete_old}: {e}")

    try:
        api.create_model(
            args.repo,
            visibility=5,  # public
            license="Apache License 2.0",
            chinese_name=args.chinese_name,
        )
        print(f"created {args.repo}")
    except Exception as e:
        # exist_ok path
        print(f"create_model note: {e}")

    api.push_model(
        model_id=args.repo,
        model_dir=str(args.dir),
        commit_message="Upload Luna Brain prototype (From-AGI-to-ASI × AIXI × 246)",
    )
    print(f"uploaded -> https://modelscope.cn/models/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
