"""Brain configs: prototype (CPU smoke) and scale_100b (memory accounting)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class BrainConfig:
    d_model: int = 256
    d_area: int = 32
    backbone_layers: int = 2
    n_actions: int = 8
    d_action: int = 16
    n_hypotheses: int = 4
    aixi_horizon: int = 3
    n_goal_slots: int = 4
    vocab_size: int = 4096
    # BriLLM/SiFu white-box speech (token=node)
    speech_vocab_size: int = 512
    d_node: int = 32
    speech_max_new: int = 8
    learning_rate: float = 1e-4
    # Nominal scale story for Pathway 1 (not loaded in prototype)
    nominal_total_params: int = 0
    active_fraction: float = 0.05
    profile: str = "prototype"


def prototype_config() -> BrainConfig:
    return BrainConfig(profile="prototype", nominal_total_params=50_000_000)


def scale_100b_config() -> BrainConfig:
    """Hundreds-of-billions nominal params; residency via sparse+quant+offload."""
    return BrainConfig(
        profile="scale_100b",
        d_model=1024,
        d_area=64,
        backbone_layers=4,
        n_actions=16,
        d_action=64,
        n_hypotheses=8,
        aixi_horizon=5,
        n_goal_slots=8,
        vocab_size=32000,
        speech_vocab_size=4096,
        d_node=64,
        speech_max_new=32,
        nominal_total_params=550_000_000_000,
        active_fraction=0.02,
    )


PROFILES = {
    "prototype": prototype_config,
    "scale_100b": scale_100b_config,
}
