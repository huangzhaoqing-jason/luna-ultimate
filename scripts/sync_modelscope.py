#!/usr/bin/env python3
"""Sync latest ModelScope luna-ultimate release metadata (and optional weights) into GitHub tree.

Usage:
  python scripts/sync_modelscope.py
  python scripts/sync_modelscope.py --download-weights
  python scripts/sync_modelscope.py --repo huang18928827157/luna-ultimate --output modelscope_release
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

DEFAULT_REPO = "huang18928827157/luna-ultimate"
API = "https://www.modelscope.cn/api/v1"
META_FILES = (
    "config.json",
    "configuration.json",
    "model.safetensors.index.json",
    "README.md",
    ".gitattributes",
)


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _download_file(repo: str, path: str, dest: Path, revision: str = "master") -> None:
    url = f"{API}/models/{repo}/repo?Revision={revision}&FilePath={path}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as resp:
        dest.write_bytes(resp.read())


def sync_metadata(repo: str, output: Path, revision: str = "master") -> dict:
    listing = _get_json(f"{API}/models/{repo}/repo/files?Recursive=true")
    files = listing.get("Data", {}).get("Files", [])
    manifest = {
        "model_id": repo,
        "source": f"https://modelscope.cn/models/{repo}",
        "github": "https://github.com/huangzhaoqing-jason/luna-ultimate",
        "revision": revision,
        "files": [],
    }
    for f in files:
        if f.get("Type") != "blob":
            continue
        manifest["files"].append(
            {
                "path": f["Path"],
                "size": f["Size"],
                "sha256": f.get("Sha256"),
                "is_lfs": bool(f.get("IsLFS")),
                "revision": f.get("Revision"),
            }
        )

    output.mkdir(parents=True, exist_ok=True)
    for name in META_FILES:
        try:
            _download_file(repo, name, output / name, revision=revision)
            print(f"synced {name}")
        except Exception as e:  # noqa: BLE001 — keep sync resilient
            print(f"skip {name}: {e}", file=sys.stderr)

    cfg = output / "config.json"
    if cfg.exists():
        try:
            manifest["preset"] = json.loads(cfg.read_text()).get("preset_name")
        except json.JSONDecodeError:
            pass

    (output / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {output / 'MANIFEST.json'} ({len(manifest['files'])} files)")
    return manifest


def download_weights(repo: str, output: Path, revision: str = "master") -> Path:
    try:
        from modelscope import snapshot_download
    except ImportError:
        print("ERROR: modelscope not installed. pip install modelscope", file=sys.stderr)
        sys.exit(1)
    local = snapshot_download(repo, revision=revision, local_dir=str(output))
    print(f"weights at {local}")
    return Path(local)


def main() -> None:
    p = argparse.ArgumentParser(description="Sync ModelScope luna-ultimate → GitHub tree")
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument("--output", type=Path, default=Path("modelscope_release"))
    p.add_argument("--revision", default="master")
    p.add_argument(
        "--download-weights",
        action="store_true",
        help="Also pull LFS weight shards into output (large; gitignored)",
    )
    args = p.parse_args()

    sync_metadata(args.repo, args.output, revision=args.revision)
    if args.download_weights:
        download_weights(args.repo, args.output, revision=args.revision)


if __name__ == "__main__":
    main()
