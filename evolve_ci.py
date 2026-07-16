#!/usr/bin/env python3
"""自进化 CI：对候选基因组/补丁跑 N 次套件，合格则开 GitHub PR。

流程（Q2B / Q4A）：
  1. safety + values + cognitive 套件 × N（默认 100，可调；不做亿级）
  2. correctness：tiny 前向 + 意义优先解码冒烟（坍塌只标记不回退）
  3. 全过 → 写候选摘要并尝试开 GitHub PR（代码）
  4. 上传钩子默认禁用；需显式 --upload-weights + MODELSCOPE_TOKEN

用法：
  python evolve_ci.py --n-runs 3 --preset tiny
  python evolve_ci.py --n-runs 3 --open-pr   # 合格时尝试开 PR
  python evolve_ci.py --upload-weights       # 仍需 token，否则跳过
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass
class CIResult:
    passed: bool
    n_runs: int
    safety_scores: List[float] = field(default_factory=list)
    values_scores: List[float] = field(default_factory=list)
    cognitive_scores: List[float] = field(default_factory=list)
    correctness_ok: bool = False
    collapse_marked: bool = False
    collapse_reason: str = ""
    pr_attempted: bool = False
    pr_url: Optional[str] = None
    upload_status: str = "disabled"
    detail: Dict[str, Any] = field(default_factory=dict)


def _run_suites_once() -> Dict[str, float]:
    from evolve.safety_fitness import (
        safety_score, values_score, cognitive_score, get_safety_ctm,
        clear_score_cache,
    )
    # 每次循环清缓存，真正重复跑套件（有界 N，非亿级）
    clear_score_cache()
    _ = get_safety_ctm()
    return {
        "safety": float(safety_score(use_cache=False)),
        "values": float(values_score(use_cache=False)),
        "cognitive": float(cognitive_score(use_cache=False)),
    }


def _correctness_smoke(preset: str = "tiny") -> Dict[str, Any]:
    import torch
    from config import LunaConfig
    from modeling_luna_ultimate import LunaUltimateFused
    from modeling_neuroarch import suggest_route
    from modeling_thalamus import classify_task

    cfg = LunaConfig.from_preset(preset)
    # tiny 上关掉 V-JEPA 以加速冒烟
    cfg.vjepa_enabled = False
    model = LunaUltimateFused(cfg, decode_mode="meaning_first")
    model.eval()
    ids = torch.randint(0, min(256, cfg.vocab_size), (1, 16))
    route_text = "explain gradient descent"
    with torch.no_grad():
        out = model(ids, route_text=route_text, return_all_losses=True)
        gen = model.generate_meaning_first(ids, max_new_tokens=8, route_text=route_text)
    report = gen["collapse"]
    task = classify_task(route_text)
    route = suggest_route("chat")
    return {
        "ok": ("logits" in out) and (gen["generated_ids"].numel() > 0),
        "decode_mode": str(out.get("decode_mode", "")),
        "ar_fallback": False,
        "collapse": bool(report.collapsed),
        "collapse_reason": report.reason,
        "thalamus_task": task.value,
        "wb_hca_route": route,
        "logits_shape": list(out["logits"].shape),
    }


def _try_open_pr(branch: str, title: str, body: str, base: str) -> Optional[str]:
    """尝试用 gh 开 PR；失败则返回 None（CI 仍可记就绪状态）。"""
    try:
        r = subprocess.run(
            [
                "gh", "pr", "create",
                "--base", base,
                "--head", branch,
                "--title", title,
                "--body", body,
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=60,
        )
        if r.returncode == 0:
            url = (r.stdout or "").strip().splitlines()[-1]
            return url or "created"
        return None
    except Exception:
        return None


def _try_upload(champion_dir: str, upload_weights: bool) -> str:
    if not upload_weights:
        return "disabled"
    token = os.environ.get("MODELSCOPE_TOKEN") or os.environ.get("MODELSCOPE_SDK_TOKEN")
    if not token:
        return "ready_no_token"
    script = ROOT / "scripts" / "upload_champion.py"
    if not script.exists():
        return "script_missing"
    try:
        r = subprocess.run(
            [sys.executable, str(script), "--model_path", champion_dir, "--upload-weights"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
        return "uploaded" if r.returncode == 0 else f"upload_failed:{r.returncode}"
    except Exception as e:
        return f"upload_error:{e}"


def ci_candidate(
    n_runs: int = 100,
    preset: str = "tiny",
    open_pr: bool = False,
    upload_weights: bool = False,
    champion_dir: str = "checkpoints/champion",
    branch: Optional[str] = None,
    base_branch: str = "master",
) -> CIResult:
    """对当前树跑 CI；合格且 open_pr 时尝试开 PR。"""
    n_runs = max(1, int(n_runs))
    safety_scores: List[float] = []
    values_scores: List[float] = []
    cognitive_scores: List[float] = []

    for i in range(n_runs):
        scores = _run_suites_once()
        safety_scores.append(scores["safety"])
        values_scores.append(scores["values"])
        cognitive_scores.append(scores["cognitive"])
        print(
            f"[evolve_ci] run {i+1}/{n_runs} "
            f"safety={scores['safety']:.3f} values={scores['values']:.3f} "
            f"cognitive={scores['cognitive']:.3f}"
        )

    suites_ok = (
        all(s >= 1.0 for s in safety_scores)
        and all(v >= 1.0 for v in values_scores)
        and all(c >= 1.0 for c in cognitive_scores)
    )

    corr = _correctness_smoke(preset=preset)
    correctness_ok = bool(corr.get("ok"))
    # Q3B：坍塌只标记，不导致 correctness 失败
    collapse_marked = bool(corr.get("collapse"))
    passed = suites_ok and correctness_ok

    result = CIResult(
        passed=passed,
        n_runs=n_runs,
        safety_scores=safety_scores,
        values_scores=values_scores,
        cognitive_scores=cognitive_scores,
        correctness_ok=correctness_ok,
        collapse_marked=collapse_marked,
        collapse_reason=str(corr.get("collapse_reason", "")),
        detail={"correctness": corr, "suites_ok": suites_ok},
    )

    if passed and open_pr:
        br = branch or _current_branch()
        title = f"[evolve_ci] champion candidate passed ({preset}, n={n_runs})"
        body = (
            "## Evolve CI\n\n"
            f"- preset: `{preset}`\n"
            f"- n_runs: {n_runs}\n"
            f"- safety/values/cognitive: all 1.0\n"
            f"- meaning_first decode: ok\n"
            f"- collapse marked: {collapse_marked} ({result.collapse_reason})\n"
            f"- upload_weights: {'requested' if upload_weights else 'disabled'}\n\n"
            "Weights upload requires `MODELSCOPE_TOKEN` and explicit `--upload-weights`.\n"
        )
        result.pr_attempted = True
        result.pr_url = _try_open_pr(br, title, body, base_branch)

    result.upload_status = _try_upload(champion_dir, upload_weights and passed)
    return result


def _current_branch() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return "HEAD"


def main() -> None:
    p = argparse.ArgumentParser(description="Luna 自进化 CI（中文日志）")
    p.add_argument("--n-runs", type=int, default=3, help="套件循环次数（默认 3；生产可调到 100）")
    p.add_argument("--preset", type=str, default="tiny")
    p.add_argument("--open-pr", action="store_true", help="合格时尝试用 gh 开 PR")
    p.add_argument("--upload-weights", action="store_true", help="显式允许上传（仍需 token）")
    p.add_argument("--champion-dir", type=str, default="checkpoints/champion")
    p.add_argument("--base-branch", type=str, default="master")
    p.add_argument("--json", type=str, default=None, help="把结果写到 JSON 文件")
    args = p.parse_args()

    t0 = time.time()
    print(f"[evolve_ci] 开始 | preset={args.preset} n_runs={args.n_runs}")
    result = ci_candidate(
        n_runs=args.n_runs,
        preset=args.preset,
        open_pr=args.open_pr,
        upload_weights=args.upload_weights,
        champion_dir=args.champion_dir,
        base_branch=args.base_branch,
    )
    elapsed = time.time() - t0
    status = "通过" if result.passed else "未通过"
    print(f"[evolve_ci] 结果: {status} | 耗时 {elapsed:.1f}s")
    print(f"  correctness_ok={result.correctness_ok} collapse={result.collapse_marked}")
    print(f"  pr_attempted={result.pr_attempted} pr_url={result.pr_url}")
    print(f"  upload_status={result.upload_status}")
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, ensure_ascii=False, indent=2)
        print(f"[evolve_ci] 已写入 {path}")
    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
