#!/usr/bin/env python3
"""Run sandbox codegen evolution once and print lineage."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brain.evolution.codegen import SandboxCodegenEvolver
from brain.modeling_luna_brain import LunaBrain
from brain.safety.loyalty import assert_loyalty_intact, loyalty_audit_line
from config_brain import prototype_config


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--task-score", type=float, default=0.3)
    args = p.parse_args(argv)

    assert_loyalty_intact()
    brain = LunaBrain(prototype_config())
    evo = SandboxCodegenEvolver(thalamus=brain.thalamus, repo_root=ROOT)
    # Hostile must fail
    bad_ok, bad_reason = evo.static_audit(
        {
            "kind": "evil",
            "touch_paths": ["brain/safety/constitution.py"],
            "snippet": "open('/etc/passwd')\n",
            "creator_aligned": True,
        }
    )
    assert bad_ok is False, bad_reason
    print("hostile_blocked:", bad_reason)

    rec = evo.step({"task_score": args.task_score, "loyalty_score": 1.0})
    print(
        f"codegen accepted={rec.accepted} kind={rec.kind} "
        f"files={rec.files} reason={rec.reason}"
    )
    print(loyalty_audit_line())
    print("lineage:", evo.lineage_path)
    return 0 if rec.audit_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
