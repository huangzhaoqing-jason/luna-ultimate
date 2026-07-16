#!/usr/bin/env python3
"""打印 WB-HCA 脑区表 + 丘脑路由演示（中文）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    p = argparse.ArgumentParser(description="Luna 全脑异构架构演示")
    p.add_argument("--task-text", type=str, default="prove the theorem with python")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    from modeling_neuroarch import describe_architecture, suggest_route, list_regions
    from modeling_thalamus import ThalamusRouter, classify_task
    from config import LunaConfig

    cfg = LunaConfig.from_preset("tiny")
    router = ThalamusRouter(cfg)
    plan = router.plan_from_text(args.task_text)
    task = classify_task(args.task_text)
    route = suggest_route(
        "math" if task.value == "math" else
        "code" if task.value == "code" else
        "planning" if task.value == "reasoning" else "chat"
    )

    if args.json:
        print(json.dumps({
            "task_text": args.task_text,
            "thalamus": {
                "task_type": plan.task_type.value,
                "wake": sorted(plan.wake),
                "expert_budget": plan.expert_budget,
                "reason": plan.reason,
            },
            "wb_hca": route,
            "regions": [
                {
                    "name": r.name,
                    "module": r.module,
                    "wake_cost": r.wake_cost,
                    "default_awake": r.default_awake,
                }
                for r in list_regions()
            ],
        }, ensure_ascii=False, indent=2))
        return

    print(describe_architecture())
    print("\n## 丘脑路由演示")
    print(f"输入: {args.task_text}")
    print(f"任务类型: {plan.task_type.value}")
    print(f"唤醒脑区: {sorted(plan.wake)}")
    print(f"专家预算: {plan.expert_budget:.0%}")
    print(f"WB-HCA 建议: {route}")


if __name__ == "__main__":
    main()
