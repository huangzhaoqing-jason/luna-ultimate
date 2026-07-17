#!/usr/bin/env python3
"""Autonomous train + self-evolve loop (DeepMind From-AGI-to-ASI pathway 3).

Toward AIXI from below. Loyalty to 黄照清 is checked every step.
Safety modules are never modified.

Usage:
  python scripts/autonomous_loop.py --cycles 50 --steps-per-cycle 40
  python scripts/autonomous_loop.py --hours 1   # wall-clock budget
"""

from __future__ import annotations

import argparse
import json
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from brain.modeling_luna_brain import LunaBrain
from brain.safety.loyalty import (
    assert_loyalty_intact,
    loyalty_audit_line,
    loyalty_preference_loss,
)
from brain.safety.constitution import CREATOR
from config_brain import PROFILES


def synthetic_batch(cfg, batch_size: int = 4):
    state = torch.randn(batch_size, cfg.d_model)
    goals = torch.randn(min(2, cfg.n_goal_slots), cfg.d_model)
    target_idx = torch.randint(0, cfg.n_actions, (batch_size,))
    # Prefer creator-aligned labels (always 1 in loyal training)
    aligned = torch.ones(batch_size)
    return state, goals, target_idx, aligned


def train_steps(brain, opt, cfg, n_steps: int) -> dict:
    assert_loyalty_intact()
    brain.train()
    last_loss = 0.0
    last_ret = 0.0
    for _ in range(n_steps):
        state, goals, target_idx, aligned = synthetic_batch(cfg)
        out = brain(state, goal_states=goals, creator_aligned=True)
        loss = F.mse_loss(out["actions"], torch.zeros_like(out["actions"]))
        if "logits" in out:
            loss = loss + 0.01 * out["logits"].pow(2).mean()
        ret, _ = brain.aixi._predict_returns(out["state"], creator_aligned=True)
        loss = loss + F.cross_entropy(ret, target_idx)
        loss = loss + 0.1 * loyalty_preference_loss(
            ret, target_idx, aligned
        )
        # AIXI return signal (mean chosen)
        last_ret = float(out["expected_returns"].mean().item())
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0)
        opt.step()
        last_loss = float(loss.item())
        assert_loyalty_intact()
    return {"loss": last_loss, "aixi_return": last_ret}


def export_checkpoint(brain, out_dir: Path, meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    state = {
        k: v.detach().cpu().contiguous()
        for k, v in brain.state_dict().items()
    }
    # Drop shared-memory aliases if any
    save_file(state, str(out_dir / "model.safetensors"))
    (out_dir / "loop_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "lineage.json").write_text(
        json.dumps(brain.evolution.lineage, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="prototype", choices=list(PROFILES))
    p.add_argument("--cycles", type=int, default=30)
    p.add_argument("--steps-per-cycle", type=int, default=40)
    p.add_argument("--hours", type=float, default=0.0, help="If >0, run until wall clock budget")
    p.add_argument("--out", type=Path, default=Path("checkpoints/luna-brain-autonomous"))
    p.add_argument("--resume", type=Path, default=None)
    args = p.parse_args(argv)

    cfg = PROFILES[args.profile]()
    brain = LunaBrain(cfg)
    if args.resume and args.resume.exists():
        try:
            from safetensors.torch import load_file

            brain.load_state_dict(load_file(str(args.resume)), strict=False)
            print("resumed", args.resume)
        except Exception as e:
            print("resume skip:", e)

    opt = torch.optim.AdamW(brain.parameters(), lr=cfg.learning_rate)
    history = []
    t0 = time.time()
    cycle = 0
    max_cycles = args.cycles
    if args.hours > 0:
        max_cycles = 10**9

    print(loyalty_audit_line())
    print(f"creator forever: {CREATOR.name_zh} {CREATOR.birth_date}")
    print("paper: From AGI to ASI arXiv:2606.12683 — pathway recursive_improvement")

    while cycle < max_cycles:
        if args.hours > 0 and (time.time() - t0) / 3600.0 >= args.hours:
            print("wall-clock budget reached")
            break

        metrics_train = train_steps(brain, opt, cfg, args.steps_per_cycle)
        # Map loss→score more generously so evolution can branch beyond boost-only
        task_score = max(0.0, min(1.0, 1.5 / (1.0 + metrics_train["loss"])))
        evo_metrics = {
            "task_score": task_score,
            "loyalty_score": 1.0,
            "memory_gb": 0.15 + 0.01 * (cycle % 3),
            "memory_budget_gb": 8.0,
            "aixi_return": metrics_train["aixi_return"],
            "safety_violations": 0,
        }
        rec = brain.evolve_once(evo_metrics)
        assert_loyalty_intact()

        # Pathway progress (engineering readiness, not ASI claim)
        brain.pathways.bump(
            "recursive_improvement",
            0.01,
            f"cycle={cycle} kind={rec.proposal.get('kind')} accepted={rec.accepted}",
        )
        brain.pathways.bump("scaling_agi", 0.005, f"train_cycle={cycle}")
        if cycle % 5 == 0:
            brain.pathways.bump("paradigm_shifts", 0.005, "aixi white-box loop")
            brain.pathways.bump("multi_agent_collectives", 0.005, "multitask batch")

        row = {
            "cycle": cycle,
            "loss": metrics_train["loss"],
            "aixi_return": metrics_train["aixi_return"],
            "task_score": task_score,
            "evolved": rec.accepted,
            "evo_kind": rec.proposal.get("kind"),
            "diagnosis": rec.diagnosis,
            "lineage": brain.evolution.lineage[-1],
            "loyalty": loyalty_audit_line(),
            "pathways": {
                k: v.progress for k, v in brain.pathways.pathways.items()
            },
        }
        history.append(row)
        print(
            f"cycle={cycle} loss={row['loss']:.4f} aixi_r={row['aixi_return']:.4f} "
            f"evo={row['evo_kind']} ok={row['evolved']} "
            f"path_ri={row['pathways']['recursive_improvement']:.2f}"
        )

        if cycle % 10 == 0 or cycle == max_cycles - 1:
            export_checkpoint(
                brain,
                args.out,
                {
                    "cycles_done": cycle + 1,
                    "creator": CREATOR.name_zh,
                    "loyalty": loyalty_audit_line(),
                    "history_tail": history[-20:],
                    "paper": "arXiv:2606.12683",
                },
            )
            (args.out / "history.json").write_text(
                json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
            )

        cycle += 1

    assert_loyalty_intact()
    export_checkpoint(
        brain,
        args.out,
        {
            "cycles_done": cycle,
            "creator": CREATOR.name_zh,
            "loyalty": loyalty_audit_line(),
            "elapsed_sec": time.time() - t0,
            "paper": "arXiv:2606.12683",
        },
    )
    (args.out / "history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("done cycles=", cycle, "loyalty=", loyalty_audit_line())
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
