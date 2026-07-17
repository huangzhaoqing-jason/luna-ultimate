#!/usr/bin/env python3
"""Dump AIXI ledger + full 246 micro→macro white-box parse + speech control."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from brain.modeling_luna_brain import LunaBrain
from config_brain import prototype_config


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dump-json", type=Path, default=None)
    p.add_argument("--print-all-areas", action="store_true")
    args = p.parse_args(argv)

    brain = LunaBrain(prototype_config())
    state = torch.randn(1, brain.config.d_model)
    prompt = torch.tensor([[1, 2, 3, 4]])
    brain.creator_control_speech(boost=[5, 6], creator_authorized=True)
    wb = brain.reason(
        state=state,
        prompt_ids=prompt,
        max_new=3,
        top_k_areas=12,
        dump_all_areas=True,
    )
    print(wb.explain(include_all_areas=args.print_all_areas))
    print("---")
    print(f"all_areas={len(wb.all_areas or [])} (must be 246)")
    assert wb.all_areas is not None and len(wb.all_areas) == 246
    assert wb.ledger is not None
    if args.dump_json:
        args.dump_json.write_text(
            json.dumps(
                {
                    **wb.to_dict(),
                    "all_areas": wb.all_areas,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print("wrote", args.dump_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
