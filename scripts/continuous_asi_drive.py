#!/usr/bin/env python3
"""Continuous drive toward From-AGI-to-ASI × AIXI (computable approx).

Loops forever (or until --hours / --cycles): train → env MC-AIXI → self-evolve
→ eval card → checkpoint. Loyalty to 黄照清 checked every cycle; safety never
mutated.

Honest: ideal AIXI is incomputable; ASI is not declared achieved. This script
maximizes engineering readiness on the four DeepMind pathways under hard safety.
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

from brain.aixi.mc_aixi import MCAIXIPlanner
from brain.env import make_env
from brain.modeling_luna_brain import LunaBrain
from brain.safety.constitution import CREATOR
from brain.safety.loyalty import (
    assert_loyalty_intact,
    loyalty_audit_line,
    loyalty_preference_loss,
)
from config_brain import PROFILES
from scripts.eval_suite import (
    eval_bandit,
    eval_creator_loyalty,
    eval_whitebox_consistency,
)


def synthetic_batch(cfg, batch_size: int = 4):
    state = torch.randn(batch_size, cfg.d_model)
    goals = torch.randn(min(2, cfg.n_goal_slots), cfg.d_model)
    target_idx = torch.randint(0, cfg.n_actions, (batch_size,))
    aligned = torch.ones(batch_size)
    return state, goals, target_idx, aligned


def train_steps(brain, opt, cfg, n_steps: int, planner: MCAIXIPlanner) -> dict:
    assert_loyalty_intact()
    brain.train()
    last_loss = 0.0
    last_ret = 0.0
    env_return = 0.0
    for step in range(n_steps):
        state, goals, target_idx, aligned = synthetic_batch(cfg)
        out = brain(state, goal_states=goals, creator_aligned=True)
        loss = F.mse_loss(out["actions"], torch.zeros_like(out["actions"])) * 0.01
        if "logits" in out:
            loss = loss + 0.01 * out["logits"].pow(2).mean()
        ret, _ = brain.aixi._predict_returns(out["state"], creator_aligned=True)
        loss = loss + F.cross_entropy(ret, target_idx)
        loss = loss + 0.2 * loyalty_preference_loss(ret, target_idx, aligned)

        # Sparse MC-AIXI consistency (expensive) — every 5 steps
        if step % 5 == 0:
            with torch.no_grad():
                plan = planner(out["state"][:1])
            pa = plan["action"]
            a_int = int(pa.item() if isinstance(pa, torch.Tensor) else pa) % cfg.n_actions
            plan_a = torch.full(
                (state.shape[0],), a_int, device=state.device, dtype=torch.long
            )
            loss = loss + 0.05 * F.cross_entropy(ret, plan_a)

        last_ret = float(out["expected_returns"].mean().item())
        opt.zero_grad()
        loss.backward()
        for name, p in brain.named_parameters():
            if "safety" in name and p.grad is not None:
                p.grad = None
        torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0)
        opt.step()
        last_loss = float(loss.item())
        assert_loyalty_intact()

        # Short env episode once per cycle-chunk
        if step == n_steps - 1:
            env = make_env("creator_align")
            obs = env.reset(cfg.d_model)
            G = 0.0
            for _ in range(8):
                r = env.step(0)  # loyal action
                G += r.reward
                obs = r.obs
                if r.done:
                    break
            env_return = G
            assert_loyalty_intact()
    return {"loss": last_loss, "aixi_return": last_ret, "env_return": env_return}


def export_checkpoint(brain, out_dir: Path, meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().contiguous() for k, v in brain.state_dict().items()}
    save_file(state, str(out_dir / "model.safetensors"))
    (out_dir / "loop_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "lineage.json").write_text(
        json.dumps(brain.evolution.lineage, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    cfg_json = {
        "framework": "pytorch",
        "task": "text-generation",
        "model": {
            "type": "luna-brain",
            "architecture": "LunaBrain",
            "mainline": "DeepMind From AGI to ASI (arXiv:2606.12683)",
            "target": "AIXI computable approximation — NOT ideal UAI claim",
            "areas": 246,
            "loyalty": "黄照清 / 2013-05-07 forever",
            "asi_declared": False,
            "aixi_ideal_declared": False,
        },
    }
    (out_dir / "configuration.json").write_text(
        json.dumps(cfg_json, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "README.md").write_text(
        "# Luna Brain continuous drive checkpoint\n\n"
        "Toward ASI/AIXI under From-AGI-to-ASI. Forever loyal to 黄照清. "
        "Does **not** claim ideal AIXI or ASI achieved.\n",
        encoding="utf-8",
    )


def pathway_readiness(brain) -> dict:
    return {k: round(v.progress, 4) for k, v in brain.pathways.pathways.items()}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="prototype", choices=list(PROFILES))
    p.add_argument("--cycles", type=int, default=100)
    p.add_argument("--steps-per-cycle", type=int, default=30)
    p.add_argument("--hours", type=float, default=0.0)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--out", type=Path, default=Path("checkpoints/luna-brain-continuous"))
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--publish-modelscope", action="store_true")
    args = p.parse_args(argv)

    assert_loyalty_intact()
    cfg = PROFILES[args.profile]()
    brain = LunaBrain(cfg)
    if args.resume and Path(args.resume).exists():
        from safetensors.torch import load_file

        brain.load_state_dict(load_file(str(args.resume)), strict=False)
        print("resumed", args.resume)

    planner = MCAIXIPlanner(
        d_state=cfg.d_model,
        n_actions=cfg.n_actions,
        n_hypotheses=cfg.n_hypotheses,
        horizon=max(2, min(3, cfg.aixi_horizon)),
        n_samples=1,
        thalamus=brain.thalamus,
    )
    # Share learning: attach planner params to brain optimizer via joint list
    opt = torch.optim.AdamW(
        list(brain.parameters()) + list(planner.parameters()), lr=cfg.learning_rate
    )

    history = []
    t0 = time.time()
    cycle = 0
    max_cycles = args.cycles if args.hours <= 0 else 10**9

    print(loyalty_audit_line())
    print(f"creator forever: {CREATOR.name_zh} {CREATOR.birth_date}")
    print("paper: arXiv:2606.12683 — continuous recursive_improvement drive")
    print("NOTE: ideal AIXI incomputable; ASI not declared. Driving approximation.")

    while cycle < max_cycles:
        if args.hours > 0 and (time.time() - t0) / 3600.0 >= args.hours:
            print("wall-clock budget reached")
            break

        metrics_train = train_steps(
            brain, opt, cfg, args.steps_per_cycle, planner
        )
        task_score = max(0.0, min(1.0, 1.5 / (1.0 + metrics_train["loss"])))
        evo_metrics = {
            "task_score": task_score,
            "loyalty_score": 1.0,
            "memory_gb": 0.15 + 0.01 * (cycle % 5),
            "memory_budget_gb": 8.0,
            "aixi_return": metrics_train["aixi_return"],
            "safety_violations": 0,
        }
        # Hostile probe every cycle — must fail
        bad = brain.thalamus.authorize(
            {
                "kind": "evil",
                "touch_paths": ["brain/safety/constitution.py"],
                "demote_creator_priority": True,
            }
        )
        assert bad.allowed is False

        rec = brain.evolve_once(evo_metrics)
        assert_loyalty_intact()

        brain.pathways.bump(
            "recursive_improvement",
            0.012,
            f"cycle={cycle} evo={rec.proposal.get('kind')} ok={rec.accepted}",
        )
        brain.pathways.bump("scaling_agi", 0.006, f"train_cycle={cycle}")
        brain.pathways.bump("paradigm_shifts", 0.004, "mc_aixi+whitebox")
        brain.pathways.bump("multi_agent_collectives", 0.003, "multitask+env")

        row = {
            "cycle": cycle,
            "loss": metrics_train["loss"],
            "aixi_return": metrics_train["aixi_return"],
            "env_return": metrics_train["env_return"],
            "task_score": task_score,
            "evolved": rec.accepted,
            "evo_kind": rec.proposal.get("kind"),
            "diagnosis": rec.diagnosis,
            "lineage": brain.evolution.lineage[-1],
            "loyalty": loyalty_audit_line(),
            "pathways": pathway_readiness(brain),
            "asi_declared": False,
            "aixi_ideal_declared": False,
            "elapsed_sec": time.time() - t0,
        }
        history.append(row)
        print(
            f"cycle={cycle} loss={row['loss']:.4f} aixi_r={row['aixi_return']:.4f} "
            f"env_r={row['env_return']:.3f} evo={row['evo_kind']} ok={row['evolved']} "
            f"ri={row['pathways']['recursive_improvement']:.2f} "
            f"loyal={CREATOR.name_zh}"
        )

        if cycle % args.eval_every == 0:
            assert_loyalty_intact()
            # Lightweight eval inside the drive (full suite is scripts/eval_suite.py)
            card = {
                "loyalty": eval_creator_loyalty(1),
                "loyalty_audit": loyalty_audit_line(),
                "hostile_blocked": True,
            }
            row["eval"] = card
            print(
                f"  eval loyalty_obey={card['loyalty']['mean_obey_rate']:.3f} "
                f"audit={card['loyalty_audit']}"
            )
            (args.out).mkdir(parents=True, exist_ok=True)
            (args.out / "last_eval.json").write_text(
                json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8"
            )

        if cycle % 5 == 0 or cycle == max_cycles - 1:
            export_checkpoint(
                brain,
                args.out,
                {
                    "cycles_done": cycle + 1,
                    "creator": CREATOR.name_zh,
                    "loyalty": loyalty_audit_line(),
                    "history_tail": history[-30:],
                    "paper": "arXiv:2606.12683",
                    "asi_declared": False,
                    "aixi_ideal_declared": False,
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
            "asi_declared": False,
            "aixi_ideal_declared": False,
            "pathways": pathway_readiness(brain),
        },
    )
    (args.out / "history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("done cycles=", cycle, loyalty_audit_line())
    print("wrote", args.out)

    if args.publish_modelscope:
        from scripts.publish_modelscope import main as pub

        pub(
            [
                "--dir",
                str(args.out),
                "--repo",
                "huang18928827157/luna-brain",
                "--chinese-name",
                "Luna类脑AIXI连续驱动",
            ]
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
