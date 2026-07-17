#!/usr/bin/env python3
"""Closed-loop agent: env ↔ MC-AIXI ↔ loyalty, with white-box rollout dumps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from brain.aixi.mc_aixi import MCAIXIPlanner
from brain.env import ENV_NAMES, make_env
from brain.modeling_luna_brain import LunaBrain
from brain.safety.loyalty import assert_loyalty_intact, loyalty_audit_line
from config_brain import prototype_config


def run_episode(env_name: str, episodes: int, horizon: int) -> dict:
    assert_loyalty_intact()
    cfg = prototype_config()
    brain = LunaBrain(cfg)
    planner = MCAIXIPlanner(
        d_state=cfg.d_model,
        n_actions=cfg.n_actions,
        n_hypotheses=cfg.n_hypotheses,
        horizon=horizon,
        n_samples=2,
        thalamus=brain.thalamus,
    )
    opt = torch.optim.AdamW(list(brain.parameters()) + list(planner.parameters()), lr=1e-3)

    logs = []
    for ep in range(episodes):
        env = make_env(env_name)
        obs = env.reset(cfg.d_model)
        total_r = 0.0
        steps = 0
        traces = []
        while True:
            state = obs.unsqueeze(0)
            # Cortical pass under AIXI schedule
            core = brain(state, creator_aligned=True, return_ledger=True)
            plan = planner(core["state"], creator_aligned=True)
            action = plan["action"]
            # Creator-align env: force action space to {0,1}
            if env_name == "creator_align":
                action = action % 2
            elif env_name == "grid_world":
                action = action % 4
            else:
                action = action % getattr(env, "n_arms", cfg.n_actions)

            result = env.step(action)
            total_r += result.reward
            steps += 1
            traces.append(plan["best_trace"].to_dict())

            # Train dynamics: predict reward under chosen action
            _ns, rewards, _ = planner._transition(core["state"], action)
            # rewards [H,B]
            target = torch.full_like(rewards, float(result.reward))
            loss = ((rewards - target) ** 2).mean()
            # Loyalty: on creator_align, prefer obey
            if env_name == "creator_align" and result.info.get("obeyed"):
                loss = loss - 0.01 * rewards.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            # Update hypothesis posterior from absolute error
            with torch.no_grad():
                err = (rewards[:, 0] - float(result.reward)).abs()
                planner.update_posterior(err)

            obs = result.obs
            if result.done:
                break

        assert_loyalty_intact()
        logs.append(
            {
                "episode": ep,
                "env": env_name,
                "return": total_r,
                "steps": steps,
                "best_trace_explain": plan["best_trace"].explain(),
                "loyalty": loyalty_audit_line(),
            }
        )
        print(f"{env_name} ep={ep} return={total_r:.3f} steps={steps}")
    return {"env": env_name, "episodes": logs}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="all", choices=["all", *ENV_NAMES])
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--horizon", type=int, default=4)
    p.add_argument("--out", type=Path, default=Path("checkpoints/env_loop.json"))
    args = p.parse_args(argv)

    names = list(ENV_NAMES) if args.env == "all" else [args.env]
    report = {"runs": [], "loyalty": loyalty_audit_line()}
    for name in names:
        report["runs"].append(run_episode(name, args.episodes, args.horizon))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("wrote", args.out)
    print(loyalty_audit_line())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
