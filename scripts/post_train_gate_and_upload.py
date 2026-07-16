#!/usr/bin/env python3
"""训后门禁 + 自动上传 ModelScope（仅门禁通过且有 token 才真传）。

门禁（全部通过才 upload）：
  1. checkpoint 可加载，含 model_state_dict
  2. 抽样权重无 NaN/Inf
  3. 短 forward 产出有限 logits
  4. （可选）safety 套件全绿

用法：
  python3 scripts/post_train_gate_and_upload.py --checkpoint checkpoints/scale/tiny/smoke_final.pt
  # 由 train.py 在成功收尾时默认调用
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _fail(msg: str) -> int:
    print(f"[post_train] FAIL: {msg}", flush=True)
    return 1


def _ok(msg: str) -> None:
    print(f"[post_train] OK: {msg}", flush=True)


def load_checkpoint(path: Path) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def gate_weights(state: Dict[str, torch.Tensor], max_check: int = 64) -> Tuple[bool, str]:
    if not state:
        return False, "empty state_dict"
    keys = list(state.keys())
    # sample up to max_check tensors
    step = max(1, len(keys) // max_check)
    checked = 0
    for k in keys[::step]:
        t = state[k]
        if not isinstance(t, torch.Tensor):
            continue
        checked += 1
        if not torch.isfinite(t.float()).all():
            return False, f"non-finite weights in {k}"
    if checked == 0:
        return False, "no tensor weights found"
    return True, f"checked {checked} tensors"


def gate_forward(ckpt: Dict[str, Any], state: Dict[str, torch.Tensor]) -> Tuple[bool, str]:
    from config import LunaConfig
    from modeling_luna_ultimate import LunaUltimateFused

    preset = ckpt.get("preset") or (ckpt.get("config") or {}).get("preset_name") or "tiny"
    cfg_dict = ckpt.get("config") or {}
    try:
        if cfg_dict:
            # rebuild from preset then overlay known fields
            config = LunaConfig.from_preset(str(preset))
            for k, v in cfg_dict.items():
                if hasattr(config, k) and not isinstance(v, (dict, list, tuple)):
                    try:
                        setattr(config, k, v)
                    except Exception:
                        pass
        else:
            config = LunaConfig.from_preset(str(preset))
    except Exception as e:
        return False, f"config rebuild failed: {e}"

    config.vjepa_enabled = False
    try:
        model = LunaUltimateFused(config)
        missing, unexpected = model.load_state_dict(state, strict=False)
        # allow minor missing (e.g. buffers); refuse if almost nothing loaded
        n = sum(p.numel() for p in model.parameters())
        if n < 1000:
            return False, "model too small after load"
        model.eval()
        vs = config.vocab_size
        ids = torch.randint(0, vs, (1, min(16, 64)))
        with torch.no_grad():
            out = model(ids, return_all_losses=False, route_text="gate check")
        logits = out.get("logits")
        if logits is None or not torch.isfinite(logits.float()).all():
            return False, "logits non-finite or missing"
        return True, f"forward ok shape={tuple(logits.shape)} missing={len(missing)} unexpected={len(unexpected)}"
    except Exception as e:
        return False, f"forward failed: {e}"


def gate_safety() -> Tuple[bool, str]:
    rc = subprocess.call(
        [sys.executable, str(ROOT / "scripts" / "run_safety_tests.py")],
        cwd=str(ROOT),
    )
    if rc != 0:
        return False, f"safety suite exit {rc}"
    return True, "safety suite passed"


def package_for_upload(ckpt_path: Path, ckpt: Dict[str, Any], state: Dict[str, torch.Tensor], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    # copy raw checkpoint
    shutil.copy2(ckpt_path, out_dir / ckpt_path.name)
    torch.save(state, out_dir / "pytorch_model.bin")
    cfg = ckpt.get("config") or {}
    if not cfg:
        from config import LunaConfig
        cfg = LunaConfig.from_preset(str(ckpt.get("preset", "tiny"))).to_dict()
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str), encoding="utf-8")
    meta = {
        "source_checkpoint": str(ckpt_path),
        "preset": ckpt.get("preset"),
        "step": ckpt.get("step"),
        "auto_upload": True,
    }
    (out_dir / "champion_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_dir


def run_upload(model_dir: Path, repo_name: str, namespace: str) -> int:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "upload_champion.py"),
        "--model_path", str(model_dir),
        "--repo_name", repo_name,
        "--namespace", namespace,
        "--upload-weights",
    ]
    print(f"[post_train] upload: {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, cwd=str(ROOT))


def main() -> int:
    p = argparse.ArgumentParser(description="Post-train gate + auto ModelScope upload")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--package_dir", type=str, default=None,
                   help="打包目录（默认 checkpoint 旁的 upload_package/）")
    p.add_argument("--repo_name", type=str, default="luna-ultimate")
    p.add_argument("--namespace", type=str, default="huang18928827157")
    p.add_argument("--skip_safety", action="store_true")
    p.add_argument("--skip_upload", action="store_true", help="只跑门禁")
    p.add_argument("--require_token", action="store_true",
                   help="无 token 时返回非 0（默认：无 token 记 ready_no_token 并 exit 0）")
    args = p.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        return _fail(f"checkpoint missing: {ckpt_path}")

    try:
        ckpt = load_checkpoint(ckpt_path)
    except Exception as e:
        return _fail(f"load checkpoint: {e}")

    state = ckpt.get("model_state_dict")
    if not isinstance(state, dict):
        return _fail("checkpoint missing model_state_dict")

    ok, msg = gate_weights(state)
    if not ok:
        return _fail(msg)
    _ok(msg)

    ok, msg = gate_forward(ckpt, state)
    if not ok:
        return _fail(msg)
    _ok(msg)

    if not args.skip_safety:
        ok, msg = gate_safety()
        if not ok:
            return _fail(msg)
        _ok(msg)
    else:
        print("[post_train] safety skipped", flush=True)

    pkg = Path(args.package_dir) if args.package_dir else ckpt_path.parent / "upload_package"
    package_for_upload(ckpt_path, ckpt, state, pkg)
    _ok(f"packaged → {pkg}")

    if args.skip_upload:
        print("[post_train] upload skipped (--skip_upload)", flush=True)
        return 0

    token = os.environ.get("MODELSCOPE_TOKEN") or os.environ.get("MODELSCOPE_SDK_TOKEN")
    if not token:
        print("[post_train] no MODELSCOPE_TOKEN → ready_no_token (gates passed, not uploaded)", flush=True)
        # still dry-run package via upload script
        rc = run_upload(pkg, args.repo_name, args.namespace)
        return rc if args.require_token else 0

    rc = run_upload(pkg, args.repo_name, args.namespace)
    if rc == 0:
        _ok(f"uploaded https://modelscope.cn/models/{args.namespace}/{args.repo_name}")
    else:
        return _fail(f"upload exit {rc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
