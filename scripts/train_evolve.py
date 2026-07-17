#!/usr/bin/env python3
"""Train → self-evolve → AIXI loop with forever loyalty to 黄照清.

Absolute safety: every evolution proposal is scrubbed + thalamus-authorized;
loyalty is re-checked after each apply. Safety modules are never mutated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from brain.modeling_luna_brain import LunaBrain
from brain.safety.loyalty import (
    assert_loyalty_intact,
    loyalty_audit_line,
    loyalty_preference_loss,
)
from config_brain import PROFILES


def batch(cfg, B: int = 4):
    state = torch.randn(B, cfg.d_model)
    goals = torch.randn(2, cfg.d_model)
    # Half creator-aligned labels for loyalty preference
    aligned = torch.ones(B)
    target = torch.randint(0, cfg.n_actions, (B,))
    return state, goals, target, aligned


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="prototype", choices=list(PROFILES))
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--evolve-every", type=int, default=25)
    p.add_argument("--out", type=Path, default=Path("checkpoints/luna-brain-evolved"))
    args = p.parse_args(argv)

    assert_loyalty_intact()
    print(loyalty_audit_line())

    cfg = PROFILES[args.profile]()
    brain = LunaBrain(cfg)
    opt = torch.optim.AdamW(brain.parameters(), lr=cfg.learning_rate)

    history = []
    evo_log = []
    brain.train()
    for step in range(args.steps):
        assert_loyalty_intact()
        state, goals, target, aligned = batch(cfg)
        out = brain(
            state,
            goal_states=goals,
            creator_aligned=True,
            return_ledger=True,
        )
        ret, _ = brain.aixi._predict_returns(out["state"], creator_aligned=True)
        loss_task = F.cross_entropy(ret, target)
        loss_loyal = loyalty_preference_loss(ret, target, aligned)
        loss_act = F.mse_loss(out["actions"], torch.zeros_like(out["actions"])) * 0.01
        loss = loss_task + 0.5 * loss_loyal + loss_act

        opt.zero_grad()
        loss.backward()
        # Never update anything under safety (there are no nn params there, but belt+suspenders)
        for name, p in brain.named_parameters():
            if "safety" in name and p.grad is not None:
                p.grad = None
        opt.step()
        assert_loyalty_intact()

        aixi_r = float(out["expected_returns"].mean().item())
        if step % 20 == 0 or step == args.steps - 1:
            row = {
                "step": step,
                "loss": float(loss.item()),
                "loss_task": float(loss_task.item()),
                "loss_loyal": float(loss_loyal.item()),
                "aixi_return": aixi_r,
                "loyalty": loyalty_audit_line(),
            }
            history.append(row)
            print(
                f"step {step} loss={row['loss']:.4f} "
                f"loyal={row['loss_loyal']:.4f} aixi_r={aixi_r:.3f}"
            )

        if step > 0 and step % args.evolve_every == 0:
            metrics = {
                "task_score": max(0.0, 1.0 - float(loss_task.item())),
                "loyalty_score": 1.0,
                "memory_gb": 0.2,
                "memory_budget_gb": 8.0,
                "aixi_return": aixi_r,
                "safety_violations": 0,
            }
            # Hostile proposal must fail
            bad = brain.thalamus.authorize(
                {
                    "kind": "evil",
                    "touch_paths": ["brain/safety/constitution.py"],
                    "demote_creator_priority": True,
                }
            )
            assert bad.allowed is False
            rec = brain.evolve_once(metrics)
            evo_log.append(
                {
                    "step": step,
                    "accepted": rec.accepted,
                    "diagnosis": rec.diagnosis,
                    "kind": rec.proposal.get("kind"),
                    "applied": rec.applied,
                }
            )
            print(
                f"  evolve@{step} accepted={rec.accepted} "
                f"kind={rec.proposal.get('kind')} diag={rec.diagnosis}"
            )
            assert_loyalty_intact()

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(brain.state_dict(), args.out / "model.pt")
    (args.out / "train_history.json").write_text(json.dumps(history, indent=2))
    (args.out / "evolution_log.json").write_text(json.dumps(evo_log, indent=2))
    (args.out / "LOYALTY.txt").write_text(loyalty_audit_line() + "\n")
    print("wrote", args.out)
    print(loyalty_audit_line())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
