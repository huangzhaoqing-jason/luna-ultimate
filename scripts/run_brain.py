#!/usr/bin/env python3
"""CLI: AIXI schedule + white-box reason + creator speech control."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from brain.modeling_luna_brain import LunaBrain
from config_brain import PROFILES


def main(argv=None):
    p = argparse.ArgumentParser(description="Run LunaBrain prototype step")
    p.add_argument("--profile", default="prototype", choices=list(PROFILES))
    p.add_argument("--evolve", action="store_true")
    args = p.parse_args(argv)

    cfg = PROFILES[args.profile]()
    brain = LunaBrain(cfg)
    state = torch.randn(1, cfg.d_model)
    goals = torch.randn(min(2, cfg.n_goal_slots), cfg.d_model)
    out = brain(state, goal_states=goals, return_schedule=True)
    print("creator:", brain.creator.name_zh, brain.creator.birth_date)
    print("activation:", tuple(out["activation"].shape))
    print("actions:", tuple(out["actions"].shape))
    print("expected_returns:", out["expected_returns"].tolist())
    print(out["schedule_trace"].explain())
    print("pathways:", brain.pathways.status()["paper"])
    print(brain.memory_report())

    wb = brain.reason(state=state, top_k_areas=6)
    print(wb.explain())

    prompt = torch.tensor([[1, 2, 3]])
    brain.creator_control_speech(force=[11, 12], creator_authorized=True)
    spoken = brain.speak(prompt, state=state, max_new=4)
    print("speech tokens:", spoken["token_ids"].tolist())
    print(spoken["explanation"])

    if args.evolve:
        rec = brain.evolve_once()
        print("evolution accepted:", rec.accepted, "diagnosis:", rec.diagnosis)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
