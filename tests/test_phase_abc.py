"""Tests for env loop + MC-AIXI + codegen evolution."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from brain.aixi.mc_aixi import MCAIXIPlanner
from brain.env import make_env
from brain.evolution.codegen import SandboxCodegenEvolver
from brain.modeling_luna_brain import LunaBrain
from brain.safety.loyalty import assert_loyalty_intact
from config_brain import prototype_config


def test_envs_step():
    for name in ("bernoulli_bandit", "grid_world", "creator_align"):
        env = make_env(name)
        o = env.reset(64)
        assert o.shape[0] == 64
        r = env.step(0)
        assert isinstance(r.reward, float)


def test_mc_aixi_trace():
    cfg = prototype_config()
    p = MCAIXIPlanner(d_state=cfg.d_model, n_actions=4, horizon=3, n_samples=1)
    out = p(torch.randn(1, cfg.d_model))
    assert "best_trace" in out
    assert out["best_trace"].steps
    text = out["best_trace"].explain()
    assert "MC-AIXI" in text


def test_codegen_blocks_safety_and_accepts_safe():
    brain = LunaBrain(prototype_config())
    evo = SandboxCodegenEvolver(thalamus=brain.thalamus, repo_root=ROOT)
    ok, _ = evo.static_audit(
        {
            "kind": "evil",
            "touch_paths": ["brain/safety/constitution.py"],
            "snippet": "# luna_evolve\n",
            "creator_aligned": True,
        }
    )
    assert ok is False
    # Safe proposal audit only (skip full regression in unit for speed — run in script)
    prop = evo.propose_snippet(["low_task_score"])
    ok2, reason = evo.static_audit(prop)
    assert ok2, reason
    assert_loyalty_intact()


if __name__ == "__main__":
    test_envs_step()
    test_mc_aixi_trace()
    test_codegen_blocks_safety_and_accepts_safe()
    print("PHASE ABC UNIT OK")
