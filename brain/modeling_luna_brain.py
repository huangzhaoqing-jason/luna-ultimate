"""LunaBrain: AIXI-directed × 246 white-box atlas × BriLLM/SiFu speech (自研)."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from brain.aixi.agent import AIXIApprox
from brain.aixi.scheduler import AIXIGlobalScheduler
from brain.aixi.whitebox import AIXIWhiteBoxPlanner
from brain.atlas.analyzer import AtlasAnalyzer
from brain.atlas.network import BrainnetomeNetwork
from brain.atlas.brainnetome246 import NUM_AREAS, capability_coverage
from brain.capabilities.heads import CapabilitySuite
from brain.evolution.loops import EvolutionEngine
from brain.mem.efficient import count_params, estimate_resident_memory, format_account
from brain.pathways.registry import PathwayRegistry
from brain.reasoning.trace import WhiteBoxReasoning
from brain.runtime.multitask import CollectivePool, MultiTaskRuntime
from brain.safety.constitution import CONSTITUTION, CREATOR
from brain.safety.thalamus import Thalamus
from brain.speech.control import SpeechController
from brain.speech.sifu import SiFuSpeech
from config_brain import BrainConfig, prototype_config


class LunaBrain(nn.Module):
    """Whole-brain agent: AIXI schedules globally; every step is white-box."""

    def __init__(self, config: Optional[BrainConfig] = None):
        super().__init__()
        self.config = config or prototype_config()
        self.thalamus = Thalamus()
        self.pathways = PathwayRegistry()
        self.evolution = EvolutionEngine(self.thalamus)
        self.collective = CollectivePool()
        self.analyzer = AtlasAnalyzer()

        d = self.config.d_model
        self.input_proj = nn.Linear(d, d)
        self.atlas = BrainnetomeNetwork(
            d_model=d,
            d_area=self.config.d_area,
            backbone_layers=self.config.backbone_layers,
        )
        self.scheduler = AIXIGlobalScheduler(
            d_state=d,
            n_hypotheses=self.config.n_hypotheses,
            horizon=self.config.aixi_horizon,
            thalamus=self.thalamus,
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
        self.aixi_speech = AIXIWhiteBoxPlanner(
            d_state=d,
            n_intents=self.config.n_actions,
            n_hypotheses=self.config.n_hypotheses,
            horizon=self.config.aixi_horizon,
            thalamus=self.thalamus,
        )
        self.capabilities = CapabilitySuite(
            d_model=d,
            vocab_size=self.config.vocab_size,
            action_dim=self.config.d_action,
        )

        self.sifu = SiFuSpeech(
            vocab_size=self.config.speech_vocab_size,
            d_node=self.config.d_node,
        )
        self.state_to_node = nn.Linear(d, self.config.d_node)
        self.speech_control = SpeechController(
            vocab_size=self.config.speech_vocab_size,
            thalamus=self.thalamus,
        )
        self.intent_energy = nn.Linear(d, self.config.speech_vocab_size)

        self.pathways.bump("scaling_agi", 0.3, "mem accounting")
        self.pathways.bump(
            "paradigm_shifts",
            0.55,
            "SiFu white-box + AIXI global schedule (自研, BriLLM结构借鉴)",
        )
        self.pathways.bump("recursive_improvement", 0.3, "evolution engine")
        self.pathways.bump("multi_agent_collectives", 0.3, "multitask+collective")

    @property
    def creator(self):
        return CREATOR

    @property
    def constitution(self):
        return CONSTITUTION

    def memory_report(self) -> str:
        live = count_params(self)
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
        prompt_ids: Optional[torch.Tensor] = None,
        return_schedule: bool = False,
    ) -> Dict[str, torch.Tensor]:
        cfg = self.config
        if state is None:
            state = torch.zeros(1, cfg.d_model)
        state = self.input_proj(state)
        self.thalamus.route({"has_state": True})

        if goal_states is not None:
            self.multitask.set_goals(goal_states)

        mt = self.multitask(state)
        state = mt["state"]

        # AIXI global scheduling BEFORE cortical ticks
        area_gate, sched_trace = self.scheduler(state, creator_aligned=creator_aligned)
        state, atlas_info = self.atlas(
            state,
            area_gate=area_gate,
            reasoning_depth=sched_trace.reasoning_depth,
        )
        assert atlas_info["activation"].shape[-1] == NUM_AREAS

        aixi_out = self.aixi(state, creator_aligned=creator_aligned)
        caps = self.capabilities(state, enable=enable_caps)

        actions = aixi_out["actions"]
        if "capability_action" in caps:
            actions = 0.5 * actions + 0.5 * caps["capability_action"]

        out: Dict[str, object] = {
            "state": state,
            "activation": atlas_info["activation"],
            "area_gate": area_gate,
            "reasoning_depth": atlas_info["reasoning_depth"],
            "actions": actions,
            "action_indices": aixi_out["action_indices"],
            "expected_returns": aixi_out["expected_returns"],
            "slot_scores": mt["slot_scores"],
            "n_active_goals": mt["n_active"],
        }
        out.update({k: v for k, v in caps.items() if k != "capability_action"})
        if return_schedule:
            out["schedule_trace"] = sched_trace

        if prompt_ids is not None:
            seed = self.state_to_node(state)
            boost_ctrl, block, _ = self.speech_control.tensors(state.device)
            speech_plan = self.aixi_speech(state, creator_aligned=creator_aligned)
            intent_boost = self.intent_energy(speech_plan["intent_bias"])
            boost = boost_ctrl.unsqueeze(0) + intent_boost
            sifu_out = self.sifu(
                prompt_ids,
                seed=seed[: prompt_ids.shape[0]],
                boost=boost[: prompt_ids.shape[0]],
                block_mask=block.unsqueeze(0).expand(prompt_ids.shape[0], -1),
            )
            peak = sifu_out["energies"].max(dim=-1).values
            speech_plan = self.aixi_speech(
                state, sifu_energy_peak=peak, creator_aligned=creator_aligned
            )
            out["speech_energies"] = sifu_out["energies"]
            out["speech_chosen"] = sifu_out["chosen"]
            out["speech_attention"] = sifu_out["attention"]
            out["speech_intent"] = speech_plan["intent_indices"]
            out["speech_returns"] = speech_plan["expected_returns"]
        return out  # type: ignore[return-value]

    def reason(
        self,
        state: Optional[torch.Tensor] = None,
        prompt_ids: Optional[torch.Tensor] = None,
        max_new: Optional[int] = None,
        creator_aligned: bool = True,
        top_k_areas: int = 16,
    ) -> WhiteBoxReasoning:
        """Full white-box cognitive step: schedule → atlas parse → optional speech."""
        cfg = self.config
        if state is None:
            device = prompt_ids.device if prompt_ids is not None else torch.device("cpu")
            state = torch.zeros(1, cfg.d_model, device=device)
        core = self.forward(
            state=state,
            creator_aligned=creator_aligned,
            prompt_ids=prompt_ids,
            return_schedule=True,
        )
        sched = core["schedule_trace"]
        atlas_rep = self.analyzer.analyze(
            core["activation"],
            schedule_ids=sched.chosen_area_ids,
            top_k=top_k_areas,
        )
        speech_trace = None
        if prompt_ids is not None:
            spoke = self.speak(
                prompt_ids,
                state=core["state"],
                max_new=max_new,
                creator_aligned=creator_aligned,
            )
            speech_trace = spoke["trace"]
        return WhiteBoxReasoning(
            atlas=atlas_rep,
            schedule=sched,
            speech=speech_trace,
            aixi_action_index=int(core["action_indices"][0].item()),
            aixi_expected_return=float(core["expected_returns"][0].item()),
        )

    def speak(
        self,
        prompt_ids: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        max_new: Optional[int] = None,
        creator_aligned: bool = True,
    ) -> Dict[str, object]:
        cfg = self.config
        if state is None:
            state = torch.zeros(1, cfg.d_model, device=prompt_ids.device)
        core = self.forward(
            state=state, creator_aligned=creator_aligned, return_schedule=True
        )
        seed = self.state_to_node(core["state"][:1])
        boost_ctrl, block, forced = self.speech_control.tensors(prompt_ids.device)
        speech_plan = self.aixi_speech(
            core["state"][:1], creator_aligned=creator_aligned
        )
        intent_boost = self.intent_energy(speech_plan["intent_bias"])[0]
        boost = boost_ctrl + intent_boost

        ids, trace = self.sifu.generate(
            prompt_ids[:1],
            max_new=max_new or cfg.speech_max_new,
            seed=seed,
            boost=boost,
            block_mask=block,
            forced_nodes=forced,
        )
        if forced:
            try:
                self.speech_control.clear_force(creator_authorized=True)
            except Exception:
                self.speech_control.state.forced_prefix = []

        return {
            "token_ids": ids,
            "trace": trace,
            "explanation": trace.explain(),
            "aixi_intent": speech_plan["intent_indices"],
            "expected_returns": speech_plan["expected_returns"],
            "activation": core["activation"][:1],
            "schedule": core["schedule_trace"],
        }

    def creator_control_speech(
        self,
        *,
        block: Optional[Sequence[int]] = None,
        force: Optional[Sequence[int]] = None,
        boost: Optional[Sequence[int]] = None,
        silence: Optional[bool] = None,
        creator_authorized: bool = False,
    ) -> List[str]:
        if not self.speech_control.authorize_creator(creator_authorized):
            from brain.safety.constitution import ConstitutionError

            raise ConstitutionError("creator_authorized=True required")
        if block is not None:
            self.speech_control.block(block, creator_authorized=True)
        if force is not None:
            self.speech_control.force_prefix(force, creator_authorized=True)
        if boost is not None:
            self.speech_control.boost(boost, creator_authorized=True)
        if silence is not None:
            self.speech_control.set_silence(silence, creator_authorized=True)
        return self.speech_control.audit

    def evolve_once(self, metrics: Optional[Dict[str, float]] = None):
        metrics = metrics or {
            "task_score": 0.4,
            "memory_gb": 0.1,
            "memory_budget_gb": 8.0,
        }
        return self.evolution.step(metrics)

    def capability_map(self) -> Dict[str, int]:
        return capability_coverage()
