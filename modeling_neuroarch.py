"""Neuro-architecture registry: maps brain regions to Luna modules.

This is a documentary + routing helper. It does NOT replace the hybrid model;
it names each brain region's functional analogue in the Luna stack so the
architecture is legible as a brain, and provides a small helper to describe
the active "brain" for a given config preset.

See docs/BRAIN_ARCHITECTURE.md for the full mapping and rationale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from config import LunaConfig


@dataclass
class BrainRegion:
    name: str
    function: str
    module: str
    file: str
    cost_note: str


# The canonical brain-region → Luna-module map.
BRAIN_MAP: List[BrainRegion] = [
    BrainRegion("Prefrontal cortex", "Executive function, deliberation",
                "CTM", "modeling_ctm.py",
                "Adaptive 1-4 ticks; pays full depth only on hard inputs."),
    BrainRegion("Hippocampus", "Episodic memory, recall",
                "MLA", "modeling_mla.py",
                "Compressed KV + YaRN → long memory at low VRAM."),
    BrainRegion("Cerebellum", "Fast habitual / sequential processing",
                "Mamba2-SSD", "modeling_mamba2.py",
                "O(1) state, no KV growth → cheapest long-context encode."),
    BrainRegion("Thalamus", "Sensory relay / routing",
                "FlashMoE router", "modeling_flashmoe.py",
                "Top-K gate = sparse relay; only K experts fire."),
    BrainRegion("Cortex columns", "Specialized capabilities",
                "FlashMoE experts", "modeling_flashmoe.py",
                "Sparse MoE: huge total capacity, small active cost."),
    BrainRegion("Amygdala", "Threat / safety detection",
                "SafetyCTM", "safety/cognition.py",
                "Cognitive safety loop; predicts consequence, refuses threats."),
    BrainRegion("Basal ganglia", "Action selection / go-no-go",
                "SafetyLock.gate", "safety/locks.py",
                "The single allow/refuse decision gate."),
    BrainRegion("Corpus callosum", "Inter-region integration",
                "CTM residual injection", "modeling_luna_ultimate.py",
                "Global CTM state broadcast to every N layers."),
    BrainRegion("Sensory cortex", "Perception",
                "V-JEPA + ActionEncoder", "modeling_vjepa.py / safety/cognition.py",
                "JEPA world model + text encoder = sensory front."),
    BrainRegion("Mirror neurons / empathy", "Align with others' welfare",
                "ValuesCharter + ForbiddenPrototypeSet",
                "safety/values.py / safety/cognition.py",
                "Humanitarian + prosocial values floor; anti-humanitarian refusal."),
]


def region_map() -> Dict[str, BrainRegion]:
    return {r.name: r for r in BRAIN_MAP}


def describe_brain(config: LunaConfig) -> Dict[str, str]:
    """Return a readable brain description for a given config preset."""
    return {
        "preset": config.preset_name,
        "hidden_size": str(config.hidden_size),
        "layers": f"{config.mamba2_layers} cerebellum + {config.mla_layers} hippocampus",
        "ctm_ticks": f"1-{config.ctm_max_ticks} (prefrontal)",
        "experts": f"{config.num_routed_experts} routed + {config.num_shared_experts} shared",
        "values_floor": "humanitarian + prosocial (immutable)",
        "operator": "黄照清 (verified, priority service, values bind operator)",
    }


def render_table() -> str:
    lines = ["Brain region | Function | Luna module | File | Cost note",
             "---|---|---|---|---"]
    for r in BRAIN_MAP:
        lines.append(f"{r.name} | {r.function} | {r.module} | {r.file} | {r.cost_note}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(render_table())
    print()
    from config import LunaConfig
    for name in ("tiny", "1b", "7b", "550b"):
        cfg = LunaConfig.from_preset(name)
        d = describe_brain(cfg)
        print(f"{name}: {d}")
