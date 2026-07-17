"""LunaBrain: From-AGI-to-ASI mainline × AIXI target × 246-area functional brain."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from brain.aixi.agent import AIXIApprox
from brain.atlas.network import BrainnetomeNetwork
from brain.atlas.brainnetome246 import NUM_AREAS, capability_coverage
from brain.capabilities.heads import CapabilitySuite
from brain.evolution.loops import EvolutionEngine
from brain.mem.efficient import count_params, estimate_resident_memory, format_account
from brain.pathways.registry import PathwayRegistry
from brain.runtime.multitask import CollectivePool, MultiTaskRuntime
from brain.safety.constitution import CONSTITUTION, CREATOR
from brain.safety.thalamus import Thalamus
from config_brain import BrainConfig, prototype_config


class LunaBrain(nn.Module):
    """Functional whole-brain agent. Not next-token-mainline; AIXI decides."""

    def __init__(self, config: Optional[BrainConfig] = None):
        super().__init__()
        self.config = config or prototype_config()
        self.thalamus = Thalamus()
        self.pathways = PathwayRegistry()
        self.evolution = EvolutionEngine(self.thalamus)
        self.collective = CollectivePool()

        d = self.config.d_model
        self.input_proj = nn.Linear(d, d)
        self.atlas = BrainnetomeNetwork(
            d_model=d,
            d_area=self.config.d_area,
            backbone_layers=self.config.backbone_layers,
        )
        self.multitask = MultiTaskRuntime(d, n_slots=self.config.n_goal_slots)
        self.aixi = AIXIApprox(
            d_state=d,
            d_action=self.config.d_action,
            n_actions=self.config.n_actions,
            n_hypotheses=self.config.n_hypotheses,
            horizon=self.config.aixi_horizon,
            thalamus=self.thalamus,
        )
        self.capabilities = CapabilitySuite(
            d_model=d,
            vocab_size=self.config.vocab_size,
            action_dim=self.config.d_action,
        )

        # Mark pathway readiness for smoke
        self.pathways.bump("scaling_agi", 0.2, "mem accounting wired")
        self.pathways.bump("paradigm_shifts", 0.2, "pluggable capability suite")
        self.pathways.bump("recursive_improvement", 0.2, "evolution engine wired")
        self.pathways.bump("multi_agent_collectives", 0.2, "multitask+collective")

    @property
    def creator(self):
        return CREATOR

    @property
    def constitution(self):
        return CONSTITUTION

    def memory_report(self) -> str:
        live = count_params(self)
        # Prototype uses live params; scale profile uses nominal story
        total = (
            self.config.nominal_total_params
            if self.config.profile == "scale_100b"
            else live
        )
        acc = estimate_resident_memory(
            total_params=total,
            active_fraction=self.config.active_fraction,
        )
        return (
            f"live_params={live:,}\n"
            f"profile={self.config.profile}\n"
            + format_account(acc)
        )

    def forward(
        self,
        state: Optional[torch.Tensor] = None,
        goal_states: Optional[torch.Tensor] = None,
        creator_aligned: bool = True,
        enable_caps: Optional[Dict[str, bool]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        state: [B, D] observation embedding (caller encodes sensors).
        goal_states: [K, D] parallel goals.
        """
        cfg = self.config
        if state is None:
            state = torch.zeros(1, cfg.d_model)
        state = self.input_proj(state)
        routed = self.thalamus.route({"has_state": True})
        _ = routed

        if goal_states is not None:
            self.multitask.set_goals(goal_states)

        mt = self.multitask(state)
        state = mt["state"]

        state, atlas_info = self.atlas(state)
        assert atlas_info["activation"].shape[-1] == NUM_AREAS

        aixi_out = self.aixi(state, creator_aligned=creator_aligned)
        caps = self.capabilities(state, enable=enable_caps)

        # Blend AIXI action with optional capability action head (not forced VLA-only)
        actions = aixi_out["actions"]
        if "capability_action" in caps:
            actions = 0.5 * actions + 0.5 * caps["capability_action"]

        out: Dict[str, torch.Tensor] = {
            "state": state,
            "activation": atlas_info["activation"],
            "actions": actions,
            "action_indices": aixi_out["action_indices"],
            "expected_returns": aixi_out["expected_returns"],
            "slot_scores": mt["slot_scores"],
            "n_active_goals": mt["n_active"],
        }
        out.update({k: v for k, v in caps.items() if k != "capability_action"})
        return out

    def evolve_once(self, metrics: Optional[Dict[str, float]] = None):
        metrics = metrics or {"task_score": 0.4, "memory_gb": 0.1, "memory_budget_gb": 8.0}
        return self.evolution.step(metrics)

    def capability_map(self) -> Dict[str, int]:
        return capability_coverage()
