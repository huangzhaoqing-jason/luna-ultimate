"""Brainnetome Atlas: 246 functional areas (capability restoration, not bio-isomorphism).

Reference: Fan et al., Cerebral Cortex 2016 — 210 cortical + 36 subcortical.
IDs 1..246; odd=left, even=right in the original atlas convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

NUM_AREAS = 246

# Macro systems used only for grouping / routing (not a substitute for 246).
MACRO_SYSTEMS: Tuple[str, ...] = (
    "thalamus_safety",
    "sensory",
    "memory_hippocampal",
    "prefrontal_executive",
    "value_amygdala",
    "basal_ganglia",
    "default_mode",
    "motor_cerebellar",
)

# Capability tags covering a full human cognitive repertoire (+ hooks for hyper abilities).
CAPABILITY_TAGS: Tuple[str, ...] = (
    "vision",
    "audition",
    "somatosensory",
    "language",
    "working_memory",
    "episodic_memory",
    "semantic_memory",
    "executive_control",
    "planning",
    "value_affect",
    "attention_gating",
    "social_cognition",
    "default_mode_imagine",
    "motor_action",
    "tool_use",
    "metacognition",
)


@dataclass(frozen=True)
class AreaSpec:
    area_id: int  # 1..246
    name: str
    macro: str
    capabilities: Tuple[str, ...]
    hemisphere: str  # "L" | "R" | "M"


def _macro_for_index(i: int) -> str:
    """Assign macros across 246 slots in stable blocks."""
    # 0-based index
    if i < 12:
        return "thalamus_safety"
    if i < 60:
        return "sensory"
    if i < 90:
        return "memory_hippocampal"
    if i < 140:
        return "prefrontal_executive"
    if i < 160:
        return "value_amygdala"
    if i < 190:
        return "basal_ganglia"
    if i < 220:
        return "default_mode"
    return "motor_cerebellar"


def _caps_for_macro(macro: str) -> Tuple[str, ...]:
    mapping = {
        "thalamus_safety": ("attention_gating", "metacognition"),
        "sensory": ("vision", "audition", "somatosensory"),
        "memory_hippocampal": ("working_memory", "episodic_memory", "semantic_memory"),
        "prefrontal_executive": ("executive_control", "planning", "language", "metacognition"),
        "value_amygdala": ("value_affect", "social_cognition"),
        "basal_ganglia": ("attention_gating", "motor_action", "tool_use"),
        "default_mode": ("default_mode_imagine", "semantic_memory", "social_cognition"),
        "motor_cerebellar": ("motor_action", "tool_use"),
    }
    return mapping[macro]


def build_area_table() -> Tuple[AreaSpec, ...]:
    areas: List[AreaSpec] = []
    for i in range(NUM_AREAS):
        area_id = i + 1
        macro = _macro_for_index(i)
        hemi = "L" if area_id % 2 == 1 else "R"
        name = f"BNA_{area_id:03d}_{macro}_{hemi}"
        areas.append(
            AreaSpec(
                area_id=area_id,
                name=name,
                macro=macro,
                capabilities=_caps_for_macro(macro),
                hemisphere=hemi,
            )
        )
    return tuple(areas)


AREA_TABLE: Tuple[AreaSpec, ...] = build_area_table()
assert len(AREA_TABLE) == NUM_AREAS


def areas_by_macro(macro: str) -> List[AreaSpec]:
    return [a for a in AREA_TABLE if a.macro == macro]


def area_index(area_id: int) -> int:
    if not 1 <= area_id <= NUM_AREAS:
        raise ValueError(f"area_id out of range: {area_id}")
    return area_id - 1


def capability_coverage() -> Dict[str, int]:
    """Count how many areas carry each capability tag."""
    counts = {c: 0 for c in CAPABILITY_TAGS}
    for a in AREA_TABLE:
        for c in a.capabilities:
            counts[c] = counts.get(c, 0) + 1
    return counts
