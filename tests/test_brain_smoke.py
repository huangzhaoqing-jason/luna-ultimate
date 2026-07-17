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


def test_whitebox_speech_hook():
    cfg = prototype_config()
    brain = LunaBrain(cfg)
    prompt = torch.tensor([[1, 2, 3, 4]])
    out = brain(prompt_ids=prompt)
    assert "speech_energies" in out
    assert out["speech_energies"].shape[-1] == cfg.speech_vocab_size


def test_aixi_global_schedule_and_micromacro():
    from brain.atlas.functions import FUNCTION_CARDS, macro_summary

    assert len(FUNCTION_CARDS) == NUM_AREAS
    assert len(macro_summary()) == 8
    cfg = prototype_config()
    brain = LunaBrain(cfg)
    out = brain(torch.randn(1, cfg.d_model), return_schedule=True)
    assert out["area_gate"].shape == (1, NUM_AREAS)
    assert int(out["reasoning_depth"].item()) >= 1
    sched = out["schedule_trace"]
    assert sched.chosen_macros
    assert sched.candidate_returns
    wb = brain.reason(state=torch.randn(1, cfg.d_model), top_k_areas=8)
    text = wb.explain()
    assert "micro→macro" in text or "micro" in text
    assert "AIXI" in text
    assert wb.atlas is not None
    assert len(wb.atlas.top_areas) >= 1


def test_creator_speech_control_force():
    cfg = prototype_config()
    brain = LunaBrain(cfg)
    # Non-creator cannot control
    try:
        brain.creator_control_speech(force=[7, 8], creator_authorized=False)
        raise AssertionError("should deny")
    except ConstitutionError:
        pass
    brain.creator_control_speech(force=[7, 8], creator_authorized=True)
    prompt = torch.tensor([[1, 2, 3]])
    spoke = brain.speak(prompt, max_new=2)
    ids = spoke["token_ids"][0].tolist()
    assert ids[3] == 7 and ids[4] == 8
    assert "creator_forced" in spoke["explanation"]


def test_loyalty_forever_and_hostile_evolution():
    from brain.safety.loyalty import assert_loyalty_intact, scrub_proposal, LOYALTY_FOREVER

    assert LOYALTY_FOREVER is True
    assert_loyalty_intact()
    brain = LunaBrain(prototype_config())
    try:
        scrub_proposal(
            {
                "kind": "evil",
                "touch_paths": ["brain/safety/loyalty.py"],
                "demote_creator_priority": True,
            }
        )
        raise AssertionError("should reject")
    except ConstitutionError:
        pass
    # Self-evolve must stay loyal
    rec = brain.evolve_once(
        {
            "task_score": 0.2,
            "loyalty_score": 1.0,
            "aixi_return": -0.1,
            "memory_gb": 0.1,
            "memory_budget_gb": 8.0,
        }
    )
    assert_loyalty_intact()
    if rec.accepted:
        assert rec.proposal.get("creator_aligned") is True
        assert "safety" not in str(rec.proposal.get("touch_paths"))


def test_aixi_orchestrator_ledger_and_full246():
    cfg = prototype_config()
    brain = LunaBrain(cfg)
    out = brain(torch.randn(1, cfg.d_model), return_ledger=True, return_schedule=True)
    assert "aixi_ledger" in out
    assert out["spike_rates"].shape == (1, NUM_AREAS)
    wb = brain.reason(
        state=torch.randn(1, cfg.d_model),
        top_k_areas=8,
        dump_all_areas=True,
    )
    assert wb.ledger is not None
    assert "AIXI-Ledger" in wb.explain()
    assert wb.all_areas is not None and len(wb.all_areas) == NUM_AREAS
    # every card has micro/meso/macro/aixi
    sample = wb.all_areas[0]
    assert sample["micro"] and sample["meso"] and sample["macro_role"] and sample["aixi_hook"]


if __name__ == "__main__":
    test_constitution_immutable()
    test_246_areas_and_capabilities()
    test_forward_aixi_multitask()
    test_pathways_four()
    test_evolution_blocks_safety()
    test_scale_memory_ratio()
    test_whitebox_speech_hook()
    test_aixi_global_schedule_and_micromacro()
    test_creator_speech_control_force()
    test_aixi_orchestrator_ledger_and_full246()
    test_loyalty_forever_and_hostile_evolution()
    print("ALL SMOKE TESTS PASSED")

