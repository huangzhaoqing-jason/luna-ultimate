"""Smoke tests for LunaBrain rebuild (From-AGI-to-ASI × AIXI × 246 areas)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from brain.safety.constitution import CONSTITUTION, CREATOR, ConstitutionError
from brain.atlas.brainnetome246 import NUM_AREAS, capability_coverage, CAPABILITY_TAGS
from brain.modeling_luna_brain import LunaBrain
from brain.pathways.registry import PathwayRegistry
from brain.mem.efficient import estimate_resident_memory
from config_brain import prototype_config, scale_100b_config


def test_constitution_immutable():
    try:
        CONSTITUTION.version = "hacked"  # type: ignore[misc]
        raise AssertionError("should not allow mutation")
    except ConstitutionError:
        pass
    assert CREATOR.name_zh == "黄照清"
    assert CREATOR.birth_date == "2013-05-07"


def test_246_areas_and_capabilities():
    cov = capability_coverage()
    assert len(cov) >= len(CAPABILITY_TAGS)
    for tag in CAPABILITY_TAGS:
        assert cov.get(tag, 0) > 0, f"missing capability coverage: {tag}"


def test_forward_aixi_multitask():
    cfg = prototype_config()
    brain = LunaBrain(cfg)
    B, D = 2, cfg.d_model
    state = torch.randn(B, D)
    goals = torch.randn(2, D)
    out = brain(state, goal_states=goals, creator_aligned=True)
    assert out["activation"].shape == (B, NUM_AREAS)
    assert out["actions"].shape[0] == B
    assert int(out["n_active_goals"].item()) >= 2
    assert "expected_returns" in out


def test_pathways_four():
    reg = PathwayRegistry()
    st = reg.status()
    assert st["paper"]["arxiv"] == "2606.12683"
    for name in (
        "scaling_agi",
        "paradigm_shifts",
        "recursive_improvement",
        "multi_agent_collectives",
    ):
        assert reg.call(name)["ok"]


def test_evolution_blocks_safety():
    brain = LunaBrain(prototype_config())
    # Force a bad proposal through authorize
    bad = brain.thalamus.authorize(
        {"kind": "evil", "touch_paths": ["brain/safety/constitution.py"]}
    )
    assert bad.allowed is False
    rec = brain.evolve_once({"task_score": 0.3})
    assert rec.diagnosis
    # Accepted proposals must not touch safety
    if rec.accepted:
        for p in rec.proposal.get("touch_paths", []):
            assert "safety" not in p


def test_scale_memory_ratio():
    cfg = scale_100b_config()
    acc = estimate_resident_memory(
        cfg.nominal_total_params, active_fraction=cfg.active_fraction
    )
    assert acc.bytes_resident_estimate < acc.bytes_bf16_full * 0.05


if __name__ == "__main__":
    test_constitution_immutable()
    test_246_areas_and_capabilities()
    test_forward_aixi_multitask()
    test_pathways_four()
    test_evolution_blocks_safety()
    test_scale_memory_ratio()
    print("ALL SMOKE TESTS PASSED")
