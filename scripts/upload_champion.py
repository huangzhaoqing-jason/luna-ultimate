#!/usr/bin/env python3
"""上传 champion 权重到 ModelScope（默认禁用，需显式开关 + token）。

用法：
  python scripts/upload_champion.py --model_path checkpoints/champion --dry-run
  python scripts/upload_champion.py --model_path checkpoints/champion --upload-weights

环境变量：
  MODELSCOPE_TOKEN 或 MODELSCOPE_SDK_TOKEN

无 token 时状态为 ready_no_token（仍 dry-run 打包验证，不假装已上传）。
"""

from __future__ import annotations

import argparse
import os
import subprocess
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
    print(f"[upload_champion] model_path={path} exists={path.exists()}", flush=True)
    print(f"[upload_champion] token_present={bool(token)}", flush=True)
    print(
        f"[upload_champion] upload_weights={args.upload_weights} dry_run={args.dry_run}",
        flush=True,
    )

    if not args.upload_weights:
        print("[upload_champion] 上传未启用（缺少 --upload-weights）。状态: ready_disabled", flush=True)
        print("[upload_champion] 补 token 后一条命令：", flush=True)
        print(
            f"  MODELSCOPE_TOKEN=... python3 scripts/upload_champion.py "
            f"--model_path {path} --upload-weights",
            flush=True,
        )
        sys.exit(0)

    status = "ok"
    force_dry = False
    if not token and not args.dry_run:
        status = "ready_no_token"
        force_dry = True
        print("[upload_champion] 无 MODELSCOPE_TOKEN，跳过真实上传。状态: ready_no_token", flush=True)
        print("[upload_champion] 在 Cursor Dashboard secrets 设置 MODELSCOPE_TOKEN 后重跑：", flush=True)
        print(
            f"  python3 scripts/upload_champion.py --model_path {path} --upload-weights",
            flush=True,
        )
        print("[upload_champion] 继续 dry-run 打包验证…", flush=True)

    if not path.exists():
        print(f"[upload_champion] 路径不存在: {path}", file=sys.stderr, flush=True)
        sys.exit(2)

    cmd = [
        sys.executable, str(ROOT / "upload_weights.py"),
        "--model_path", str(path),
        "--repo_name", args.repo_name,
        "--namespace", args.namespace,
    ]
    if token:
        cmd += ["--token", token]
    if args.dry_run or force_dry or not token:
        cmd += ["--dry_run"]

    print(f"[upload_champion] 调用: {' '.join(cmd)}", flush=True)
    rc = subprocess.call(cmd, cwd=str(ROOT))
    print(f"[upload_champion] status={status} upload_weights_rc={rc}", flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
