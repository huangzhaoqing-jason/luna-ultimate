"""246 Brainnetome areas: micro → meso → macro functional catalog (white-box).

Not a biological synapse clone — each area has an explicit, inspectable role
at three scales so AIXI scheduling and reasoning are never a black box.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from brain.atlas.brainnetome246 import AREA_TABLE, MACRO_SYSTEMS, NUM_AREAS, AreaSpec


@dataclass(frozen=True)
class ScaleRole:
    """One scale of explanation for an area."""

    scale: str  # micro | meso | macro
    description: str


@dataclass(frozen=True)
class AreaFunctionCard:
    area_id: int
    name: str
    macro: str
    hemisphere: str
    capabilities: Tuple[str, ...]
    micro: str
    meso: str
    macro_role: str
    aixi_hook: str  # how AIXI uses this area in global scheduling


# Meso narrative templates per macro system (self-developed; BriLLM-inspired transparency).
_MESO: Dict[str, str] = {
    "thalamus_safety": (
        "Relays and gates cross-system traffic; hard safety checks before any "
        "self-modification or high-impact action reaches effectors."
    ),
    "sensory": (
        "Builds sparse multimodal features (vision/audition/somatosensory) into "
        "shared state; feeds predictive hypotheses without opaque attention soup."
    ),
    "memory_hippocampal": (
        "Binds episodes, compresses working traces, retrieves semantic cues for "
        "AIXI hypothesis scoring over finite horizons."
    ),
    "prefrontal_executive": (
        "Runs multi-tick plans, metacognitive self-checks, and white-box goal "
        "decomposition; primary seat of deliberative AIXI expectimax depth."
    ),
    "value_amygdala": (
        "Scores affective/value salience; permanently boosts creator-aligned "
        "rewards under the read-only constitution."
    ),
    "basal_ganglia": (
        "Selects which skills/areas get compute (action gating); implements "
        "AIXI's discrete action/area Top-K scheduling."
    ),
    "default_mode": (
        "Offline imagination / counterfactual rollouts when external drive is low; "
        "extends AIXI horizon via simulated trajectories."
    ),
    "motor_cerebellar": (
        "Emits timed action/speech motor commands; SiFu white-box nodes are the "
        "readable effector path for language."
    ),
}

_MACRO_ROLE: Dict[str, str] = {
    "thalamus_safety": "Global bus + safety lock (highest privilege).",
    "sensory": "World interface — perception → state.",
    "memory_hippocampal": "Memory substrate for hypotheses.",
    "prefrontal_executive": "Deliberation / reasoning core.",
    "value_amygdala": "Loyalty & value weighting.",
    "basal_ganglia": "Compute & action scheduler.",
    "default_mode": "Internal simulation / research mode.",
    "motor_cerebellar": "Effector: action + speech.",
}

_AIXI_HOOK: Dict[str, str] = {
    "thalamus_safety": "Constraint filter on every AIXI proposal.",
    "sensory": "Supplies observation e_t for AIXI history.",
    "memory_hippocampal": "Stores/retrieves context for ρ hypotheses.",
    "prefrontal_executive": "Hosts expectimax depth / hypothesis mix.",
    "value_amygdala": "Shapes reward r_t (creator priority).",
    "basal_ganglia": "Chooses which area-blocks get ticks (schedule a).",
    "default_mode": "Rolls out imaginary futures under ρ.",
    "motor_cerebellar": "Executes chosen a_t (incl. SiFu speech).",
}


def _micro_line(spec: AreaSpec) -> str:
    caps = ",".join(spec.capabilities)
    # Unique per-area micro story (column id, hemi, macro slot, caps)
    slot = (spec.area_id - 1) % 17
    return (
        f"BNA#{spec.area_id:03d}/{spec.hemisphere} macro={spec.macro} slot={slot}: "
        f"LIF-lite column (leak·V→σ-spike) on shared backbone; "
        f"residual amp∝gate; caps=[{caps}]; "
        f"readable weights in AreaColumn.in_proj/out_proj."
    )


def build_function_cards() -> Tuple[AreaFunctionCard, ...]:
    cards: List[AreaFunctionCard] = []
    for spec in AREA_TABLE:
        cards.append(
            AreaFunctionCard(
                area_id=spec.area_id,
                name=spec.name,
                macro=spec.macro,
                hemisphere=spec.hemisphere,
                capabilities=spec.capabilities,
                micro=_micro_line(spec),
                meso=_MESO[spec.macro],
                macro_role=_MACRO_ROLE[spec.macro],
                aixi_hook=_AIXI_HOOK[spec.macro],
            )
        )
    assert len(cards) == NUM_AREAS
    return tuple(cards)


FUNCTION_CARDS: Tuple[AreaFunctionCard, ...] = build_function_cards()


def card_for(area_id: int) -> AreaFunctionCard:
    if not 1 <= area_id <= NUM_AREAS:
        raise ValueError(area_id)
    return FUNCTION_CARDS[area_id - 1]


def macro_summary() -> Dict[str, Dict[str, str]]:
    """One entry per macro system with meso/macro/AIXI hooks."""
    out = {}
    for m in MACRO_SYSTEMS:
        out[m] = {
            "meso": _MESO[m],
            "macro_role": _MACRO_ROLE[m],
            "aixi_hook": _AIXI_HOOK[m],
            "n_areas": sum(1 for c in FUNCTION_CARDS if c.macro == m),
        }
    return out
