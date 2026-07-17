"""White-box SiFu speech + creator control tests."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from brain.safety.constitution import ConstitutionError
from brain.modeling_luna_brain import LunaBrain
from config_brain import prototype_config


def test_sifu_trace_is_white_box():
    brain = LunaBrain(prototype_config())
    prompt = torch.tensor([[1, 2, 3]])
    out = brain.speak(prompt, max_new=4)
    trace = out["trace"]
    assert len(trace.steps) == 4
    assert trace.paradigm.startswith("SiFu")
    assert out["token_ids"].shape[1] == 3 + 4
    # Every step exposes candidate energies
    for s in trace.steps:
        assert s.candidate_top
        assert isinstance(s.chosen_node, int)


def test_creator_force_speech():
    brain = LunaBrain(prototype_config())
    brain.creator_control_speech(force=[7, 8, 9], creator_authorized=True)
    prompt = torch.tensor([[1, 2]])
    out = brain.speak(prompt, max_new=3)
    # First three new tokens must be forced
    assert out["token_ids"][0, 2:].tolist() == [7, 8, 9]
    assert all(s.note == "creator_forced" for s in out["trace"].steps)


def test_non_creator_cannot_control():
    brain = LunaBrain(prototype_config())
    try:
        brain.creator_control_speech(force=[1], creator_authorized=False)
        raise AssertionError("should deny")
    except ConstitutionError:
        pass


def test_block_nodes():
    brain = LunaBrain(prototype_config())
    # Boost 5 heavily and block everything else via silence sink path — simpler: block 5
    brain.creator_control_speech(block=[5], boost=[6], creator_authorized=True)
    prompt = torch.tensor([[1, 2, 3]])
    out = brain.forward(prompt_ids=prompt)
    assert out["speech_energies"][0, 5] < out["speech_energies"][0, 6]


if __name__ == "__main__":
    test_sifu_trace_is_white_box()
    test_creator_force_speech()
    test_non_creator_cannot_control()
    test_block_nodes()
    print("WHITEBOX SPEECH TESTS PASSED")
