#!/usr/bin/env python3
"""tiny 冒烟：JEPA 总控改变 ticks/budget；对照 heuristic。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import LunaConfig
from modeling_luna_ultimate import LunaUltimateFused


def _run(model, text: str):
    ids = torch.randint(0, model.config.vocab_size, (1, 32))
    # 用可复现伪 tokenize：字符哈希填前缀
    for i, c in enumerate(text[:16]):
        ids[0, i] = (ord(c) * 131 + 7) % model.config.vocab_size
    with torch.no_grad():
        out = model(ids, route_text=text, return_all_losses=False)
    return {
        "text": text,
        "jepa_driven": bool(out.get("jepa_driven")),
        "jepa_uncertainty": out.get("jepa_uncertainty"),
        "jepa_compute_scale": out.get("jepa_compute_scale"),
        "ctm_ticks_plan": out.get("ctm_ticks_plan"),
        "expert_budget": float(out["expert_budget"]) if out.get("expert_budget") is not None else None,
        "thalamus_task": out.get("thalamus_task"),
        "thalamus_reason": out.get("thalamus_reason"),
        "avg_ticks": float(out["avg_ticks"]) if out.get("avg_ticks") is not None else None,
    }


def main() -> int:
    cfg = LunaConfig.from_preset("tiny")
    model = LunaUltimateFused(cfg)
    model.eval()

    simple = _run(model, "hi")
    hard = _run(model, "prove the theorem by induction and design a complex architecture strategy")

    # heuristic 对照
    cfg_h = LunaConfig.from_preset("tiny", route_mode="heuristic")
    model_h = LunaUltimateFused(cfg_h)
    model_h.eval()
    hard_h = _run(model_h, "prove the theorem by induction and design a complex architecture strategy")

    ticks_up = (hard["ctm_ticks_plan"] or 0) >= (simple["ctm_ticks_plan"] or 0)
    budget_up = (hard["expert_budget"] or 0) > (simple["expert_budget"] or 0)
    report = {
        "jepa_simple": simple,
        "jepa_hard": hard,
        "heuristic_hard": hard_h,
        "ok_jepa_driven": simple["jepa_driven"] and hard["jepa_driven"],
        "ok_heuristic_off": not hard_h["jepa_driven"],
        "ok_fields": (
            simple["ctm_ticks_plan"] is not None
            and hard["ctm_ticks_plan"] is not None
            and simple["jepa_uncertainty"] is not None
        ),
        "ok_hard_ge_simple_ticks": ticks_up,
        "ok_hard_gt_simple_budget": budget_up,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not (
        report["ok_jepa_driven"]
        and report["ok_heuristic_off"]
        and report["ok_fields"]
        and report["ok_hard_ge_simple_ticks"]
        and report["ok_hard_gt_simple_budget"]
    ):
        print("SMOKE FAIL", file=sys.stderr)
        return 1
    print("SMOKE OK: JEPA control wires ticks/budget/uncertainty")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
