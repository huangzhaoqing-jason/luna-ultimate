"""Whole-Brain Hybrid Cognitive Architecture (WB-HCA) mapping.

Maps biological brain regions onto Luna modules. This is an engineering
analogy for modularity and interpretability — not a claim of biological
equivalence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence


@dataclass(frozen=True)
class BrainRegion:
    name: str
    biological_analogue: str
    module: str
    function: str
    wake_cost: str  # low | mid | high
    default_awake: bool


# Canonical WB-HCA table (Q1B meaning-first decoding; no AR fallback).
WB_HCA: Dict[str, BrainRegion] = {
    "prefrontal_cortex": BrainRegion(
        name="prefrontal_cortex",
        biological_analogue="prefrontal cortex / executive control",
        module="safety.SafetyCTM + MeaningPlanner",
        function="long-horizon planning, values gate, refuse/allow before act",
        wake_cost="mid",
        default_awake=True,
    ),
    "thalamus": BrainRegion(
        name="thalamus",
        biological_analogue="thalamus / gating hub",
        module="modeling_thalamus.ThalamusRouter",
        function="task-type routing, expert budget, on-demand wake",
        wake_cost="low",
        default_awake=True,
    ),
    "hippocampus": BrainRegion(
        name="hippocampus",
        biological_analogue="hippocampus / episodic memory",
        module="modeling_hippocampus.EpisodicMemory + rag/",
        function="retrieve prior episodes; write durable memories",
        wake_cost="mid",
        default_awake=False,
    ),
    "basal_ganglia": BrainRegion(
        name="basal_ganglia",
        biological_analogue="basal ganglia / action selection",
        module="modeling_flashmoe.FlashMoE",
        function="sparse expert selection under thalamus budget",
        wake_cost="high",
        default_awake=False,
    ),
    "cerebellum": BrainRegion(
        name="cerebellum",
        biological_analogue="cerebellum / predictive control",
        module="modeling_cerebellum.CerebellarCorrector",
        function="error prediction and fine motor / output correction",
        wake_cost="mid",
        default_awake=False,
    ),
    "parietal": BrainRegion(
        name="parietal",
        biological_analogue="parietal lobe / spatial-math",
        module="modeling_parietal.ParietalReasoner",
        function="spatial, numeric, and structured reasoning",
        wake_cost="mid",
        default_awake=False,
    ),
    "temporal_association": BrainRegion(
        name="temporal_association",
        biological_analogue="temporal association cortex",
        module="modeling_vjepa.VJEPAEncoder",
        function="cross-modal meaning targets for JEPA",
        wake_cost="high",
        default_awake=False,
    ),
    "language_areas": BrainRegion(
        name="language_areas",
        biological_analogue="Broca/Wernicke analogues",
        module="MeaningFirstDecoder (no AR fallback)",
        function="decode JEPA meaning vectors into tokens",
        wake_cost="mid",
        default_awake=True,
    ),
    "working_memory": BrainRegion(
        name="working_memory",
        biological_analogue="working memory / persistent state",
        module="modeling_mamba2.Mamba2Block",
        function="long-context state compression",
        wake_cost="mid",
        default_awake=True,
    ),
    "attention_networks": BrainRegion(
        name="attention_networks",
        biological_analogue="frontoparietal attention",
        module="modeling_mla.MLA",
        function="selective focus with compressed KV",
        wake_cost="mid",
        default_awake=True,
    ),
    "default_mode": BrainRegion(
        name="default_mode",
        biological_analogue="default mode network",
        module="evolve/ + code_evolve/ (offline)",
        function="self-reflection, architecture search, offline rehearsal",
        wake_cost="high",
        default_awake=False,
    ),
    "amygdala_safety": BrainRegion(
        name="amygdala_safety",
        biological_analogue="amygdala / threat detection (engineering analogue)",
        module="safety/charter + values + locks",
        function="hard floors: refuse harm / anti-humanitarian / disloyalty patterns",
        wake_cost="low",
        default_awake=True,
    ),
}


TASK_REGION_BIAS: Dict[str, Sequence[str]] = {
    "chat": ("language_areas", "working_memory", "attention_networks", "prefrontal_cortex"),
    "code": ("language_areas", "cerebellum", "basal_ganglia", "working_memory"),
    "math": ("parietal", "cerebellum", "prefrontal_cortex", "working_memory"),
    "vision": ("temporal_association", "parietal", "attention_networks"),
    "planning": ("prefrontal_cortex", "hippocampus", "working_memory", "default_mode"),
    "memory": ("hippocampus", "working_memory", "language_areas"),
    "safety": ("amygdala_safety", "prefrontal_cortex"),
}


def list_regions() -> List[BrainRegion]:
    return list(WB_HCA.values())


def regions_for_task(task: str) -> List[BrainRegion]:
    keys = TASK_REGION_BIAS.get(task, TASK_REGION_BIAS["chat"])
    return [WB_HCA[k] for k in keys if k in WB_HCA]


def awake_modules(task: str) -> List[str]:
    """Modules that should be woken for a task (plus always-on defaults)."""
    awake = {r.module for r in WB_HCA.values() if r.default_awake}
    for r in regions_for_task(task):
        awake.add(r.module)
    return sorted(awake)


def describe_architecture() -> str:
    lines = ["# WB-HCA Region Map", ""]
    for r in list_regions():
        lines.append(
            f"- **{r.name}** ← {r.biological_analogue}\n"
            f"  - module: `{r.module}`\n"
            f"  - function: {r.function}\n"
            f"  - wake_cost: {r.wake_cost}; default_awake: {r.default_awake}"
        )
    return "\n".join(lines)


def suggest_route(task: Optional[str] = None) -> Dict[str, object]:
    task = task or "chat"
    regs = regions_for_task(task)
    return {
        "task": task,
        "regions": [r.name for r in regs],
        "modules": awake_modules(task),
        "decode_mode": "meaning_first",
        "ar_fallback": False,
    }
