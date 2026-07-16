#!/usr/bin/env python3
"""上传 champion 权重到 ModelScope（默认禁用，需显式开关 + token）。

用法：
  python scripts/upload_champion.py --model_path checkpoints/champion --dry-run
  python scripts/upload_champion.py --model_path checkpoints/champion --upload-weights

环境变量：
  MODELSCOPE_TOKEN 或 MODELSCOPE_SDK_TOKEN
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    p = argparse.ArgumentParser(description="上传 Luna champion 到 ModelScope")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--repo_name", type=str, default="luna-champion")
    p.add_argument("--namespace", type=str, default="huang18928827157")
    p.add_argument("--upload-weights", action="store_true",
                   help="显式允许上传；缺省只做就绪检查")
    p.add_argument("--dry-run", action="store_true", help="只准备文件不上传")
    args = p.parse_args()

    token = os.environ.get("MODELSCOPE_TOKEN") or os.environ.get("MODELSCOPE_SDK_TOKEN")
    path = Path(args.model_path)
    print(f"[upload_champion] model_path={path} exists={path.exists()}")
    print(f"[upload_champion] token_present={bool(token)}")
    print(f"[upload_champion] upload_weights={args.upload_weights} dry_run={args.dry_run}")

    if not args.upload_weights:
        print("[upload_champion] 上传未启用（缺少 --upload-weights）。状态: ready_disabled")
        sys.exit(0)

    if not token and not args.dry_run:
        print("[upload_champion] 无 MODELSCOPE_TOKEN，跳过上传。状态: ready_no_token")
        sys.exit(0)

    if not path.exists():
        print(f"[upload_champion] 路径不存在: {path}", file=sys.stderr)
        sys.exit(2)

    # 复用现有 upload_weights.py CLI
    cmd = [
        sys.executable, str(ROOT / "upload_weights.py"),
        "--model_path", str(path),
        "--repo_name", args.repo_name,
        "--namespace", args.namespace,
    ]
    if token:
        cmd += ["--token", token]
    if args.dry_run or not token:
        cmd += ["--dry_run"]

    print(f"[upload_champion] 调用: {' '.join(cmd[:6])} ...")
    os.execv(sys.executable, cmd)


if __name__ == "__main__":
    main()
