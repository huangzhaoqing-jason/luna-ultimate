"""Luna-Ultimate Fused: hybrid CTM × Mamba2 × MLA × FlashMoE × JEPA.

支持 decode_mode:
  - "meaning_first"（默认，Q1B）：MeaningPlanner → MeaningFirstDecoder，无 AR fallback
  - "autoregressive"：仅训练稳定性对照，不作为推理回退路径（Q3B）
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import LunaConfig, count_parameters, verify_parameters
from loss_manager import StageAwareLossManager
from modeling_ctm import CTM
from modeling_flashmoe import FlashMoE
from modeling_mamba2 import Mamba2Block
from modeling_jepa_control import ControlSignals, JEPAController
from modeling_meaning import CollapseDetector, MeaningFirstDecoder, MeaningPlanner
from modeling_mla import MLABlock
from modeling_thalamus import RoutePlan, ThalamusRouter
from quant_utils import RMSNorm


class LunaUltimateFused(nn.Module):
    """Fused Luna model with block-level CTM injection."""

    def __init__(self, config: LunaConfig, decode_mode: str = "meaning_first"):
        super().__init__()
        self.config = config
        self.d_model = config.hidden_size
        self.vocab_size = config.vocab_size
        self.num_layers = config.num_hidden_layers
        self.mamba2_layers = config.mamba2_layers
        self.mla_layers = config.mla_layers
        self.ctm_inject_every = getattr(config, "ctm_inject_every", 4)
        self._grad_ckpt = False
        if decode_mode not in ("meaning_first", "autoregressive"):
            raise ValueError(f"unknown decode_mode={decode_mode!r}")
        self.decode_mode = decode_mode

        self.embed_tokens = nn.Embedding(config.vocab_size, self.d_model)

        self.vjepa_enabled = getattr(config, "vjepa_enabled", True)
        if self.vjepa_enabled:
            from modeling_vjepa import VJEPA
            vjepa_config = getattr(config, "vjepa_config", {})
            self.vjepa = VJEPA(
                img_size=vjepa_config.get("img_size", (224, 224)),
                patch_size=vjepa_config.get("patch_size", (2, 16, 16)),
                in_channels=vjepa_config.get("in_channels", 3),
                embed_dim=vjepa_config.get("embed_dim", 1024),
                encoder_depth=vjepa_config.get("encoder_depth", 24),
                predictor_depth=vjepa_config.get("predictor_depth", 6),
                num_heads=vjepa_config.get("num_heads", 16),
                mask_ratio=vjepa_config.get("mask_ratio", 0.75),
                use_target_encoder=vjepa_config.get("use_target_encoder", True),
                ema_decay=vjepa_config.get("ema_decay", 0.996),
                out_dim=self.d_model,
            )
        else:
            self.vjepa = None

        self.ctm = CTM(config)
        self.mamba_blocks = nn.ModuleList([
            Mamba2Block(config, layer_idx=i) for i in range(self.mamba2_layers)
        ])
        self.mla_blocks = nn.ModuleList([
            MLABlock(config, layer_idx=i + self.mamba2_layers)
            for i in range(self.mla_layers)
        ])
        self.moe_layers = nn.ModuleList([
            FlashMoE(config) for _ in range(self.num_layers)
        ])

        self.final_norm = RMSNorm(self.d_model, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

        # WB-HCA: 丘脑路由 + 意义优先解码（Q1B，无 fallback）
        self.thalamus = ThalamusRouter(config)
        self.meaning_planner = MeaningPlanner(config)
        self.meaning_decoder = MeaningFirstDecoder(config, lm_head=self.lm_head)
        self.collapse_detector = CollapseDetector()
        self.last_route_plan: Optional[RoutePlan] = None
        self.last_jepa_signals: Optional[ControlSignals] = None

        # JEPA 总控：驱动丘脑 / ticks / MoE / Meaning / Motor / WA
        self.jepa_control_enabled = bool(getattr(config, "jepa_control_enabled", True))
        if self.jepa_control_enabled:
            self.jepa_controller = JEPAController(config, ctm=self.ctm)
        else:
            self.jepa_controller = None

        # 脑区模块（按需接入；tiny 上轻量）
        from modeling_cerebellum import CerebellarCorrector
        from modeling_hippocampus import HippocampusModule
        from modeling_parietal import ParietalReasoner
        self.cerebellum = CerebellarCorrector(config)
        self.hippocampus = HippocampusModule(config)
        self.parietal = ParietalReasoner(config)
        # 颞叶 RAG（内存占位）
        try:
            from rag import build_rag
            self.rag = build_rag("memory")
        except Exception:
            self.rag = None
        # 默认灌几条知识占位
        if self.rag is not None and len(self.rag) == 0:
            for t in [
                "gradient descent minimizes loss",
                "mamba2 ssd is o(1) state",
                "git commit records a snapshot",
                "lora low rank adaptation finetune",
            ]:
                self.rag.add(t)

        # 运动动作中枢（M1/premotor/SMA）— 延迟导入避免循环
        self.motor_enabled = bool(getattr(config, "motor_enabled", True))
        if self.motor_enabled:
            from modeling_motor import MotorCortex
            self.motor = MotorCortex(config)
        else:
            self.motor = None

        # VLA / World-Action（可选能力面；默认开启轻量头）
        self.vla_enabled = bool(getattr(config, "vla_enabled", True))
        self.wa_enabled = bool(getattr(config, "wa_enabled", True))
        if self.vla_enabled:
            from modeling_vla import ActionHead
            self.action_head = ActionHead(config)
        else:
            self.action_head = None
        if self.wa_enabled:
            from modeling_world_action import WorldActionModule
            n_act = len(self.action_head.tokenizer) if self.action_head is not None else 10
            self.world_action = WorldActionModule(config, n_actions=n_act)
        else:
            self.world_action = None

        self.skip_probe_12 = nn.Linear(config.ctm_n_neurons, 1, bias=False)
        self.skip_probe_24 = nn.Linear(config.ctm_n_neurons, 1, bias=False)

        self.loss_manager = StageAwareLossManager(stage=1, use_uncertainty=False)
        self.training_stage = 1
        self.last_avg_ticks = 0.0

        if config.preset_name in ("550b", "77b_active"):
            self._verify_and_print_params()

    def _verify_and_print_params(self):
        verify_parameters(self.config)

    def gradient_checkpointing_enable(self, **kwargs):
        self._grad_ckpt = True

    def gradient_checkpointing_disable(self):
        self._grad_ckpt = False

    def set_training_stage(self, stage: int):
        """Stage 1: text LM; Stage 2: +CTM-JEPA; Stage 3: +V-JEPA."""
        self.training_stage = stage
        self.loss_manager.set_stage(stage)

        for name, param in self.named_parameters():
            param.requires_grad = True

        if stage in (1, 2) and self.vjepa is not None:
            for name, param in self.named_parameters():
                if "vjepa" in name:
                    param.requires_grad = False

    def _should_skip_layers(
        self, ctm_state: torch.Tensor, layer_idx: int
    ) -> torch.Tensor:
        if layer_idx == min(12, self.num_layers // 2):
            probe = self.skip_probe_12(ctm_state)
        elif layer_idx == min(24, self.num_layers - 1):
            probe = self.skip_probe_24(ctm_state)
        else:
            return torch.zeros(
                ctm_state.shape[0], dtype=torch.bool, device=ctm_state.device
            )
        return torch.sigmoid(probe.squeeze(-1)) > self.config.layer_skip_prob_threshold

    def _refresh_ctm(
        self,
        hidden_states: torch.Tensor,
        use_ctm_adaptive: bool,
        return_jepa: bool,
        tick_list: List[int],
        ctmj_acc: torch.Tensor,
    ):
        ctm_output, _, ticks, ctmj = self.ctm(
            hidden_states,
            use_adaptive_early_exit=use_ctm_adaptive,
            return_jepa_loss=return_jepa,
        )
        tick_list.append(ticks)
        if ctmj is not None:
            ctmj_acc = ctmj_acc + ctmj
        ctm_state = self.ctm.get_ctm_state(hidden_states)
        return ctm_output, ctm_state, ctmj_acc

    def _apply_expert_budget(self, plan: Optional[RoutePlan]) -> List[int]:
        """按丘脑预算临时缩放 MoE top_k；返回旧值以便恢复。"""
        prev = [int(m.top_k) for m in self.moe_layers]
        if plan is None:
            return prev
        k = self.thalamus.expert_top_k(self.config, plan)
        for m in self.moe_layers:
            m.top_k = k
        return prev

    def _restore_expert_budget(self, prev: List[int]) -> None:
        for m, k in zip(self.moe_layers, prev):
            m.top_k = k

    def _apply_route(self, plan: Optional[RoutePlan]) -> Dict[str, object]:
        """丘脑真调度：按 plan 设 CTM ticks / MoE top_k / layer-skip 等。

        返回旧值字典，forward 结束后恢复。这是「低算力」核心路径。
        """
        prev: Dict[str, object] = {"moe_topk": [int(m.top_k) for m in self.moe_layers]}
        if plan is None:
            prev["ctm_max_ticks"] = self.ctm.max_ticks
            prev["layer_skip_prob"] = self.config.layer_skip_prob_threshold
            return prev
        # MoE top_k
        k = self.thalamus.expert_top_k(self.config, plan)
        for m in self.moe_layers:
            m.top_k = k
        # CTM ticks 上限（运行时覆盖，不动 config 字段）
        prev["ctm_max_ticks"] = self.ctm.max_ticks
        self.ctm.max_ticks = max(1, min(plan.ctm_ticks, self.config.ctm_max_ticks))
        # layer-skip 阈值（简单任务更激进跳层）
        prev["layer_skip_prob"] = self.config.layer_skip_prob_threshold
        if plan.enable_layer_skip:
            self.config.layer_skip_prob_threshold = 0.3
        else:
            self.config.layer_skip_prob_threshold = 0.5
        return prev

    def _restore_route(self, prev: Dict[str, object]) -> None:
        for m, k in zip(self.moe_layers, prev["moe_topk"]):
            m.top_k = k
        self.ctm.max_ticks = int(prev["ctm_max_ticks"])
        self.config.layer_skip_prob_threshold = prev["layer_skip_prob"]

    def forward(
        self,
        input_ids: torch.Tensor,
        visual_input: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_ctm_adaptive: bool = True,
        use_dynamic_skip: bool = False,
        return_all_losses: bool = False,
        use_int4_cache: Optional[bool] = None,
        operator_embedding: Optional[torch.Tensor] = None,
        route_text: Optional[str] = None,
        decode_mode: Optional[str] = None,
        labels: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        next_visual_hidden: Optional[torch.Tensor] = None,
        compute_vla: bool = False,
        compute_wa: bool = False,
        action_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        del attention_mask  # reserved
        B, L_txt = input_ids.shape
        device = input_ids.device
        outputs: Dict[str, Any] = {}
        mode = decode_mode or self.decode_mode
        if use_int4_cache is None:
            use_int4_cache = bool(self.config.use_kv_cache_int4) and not self.training

        jepa_on = (
            self.jepa_control_enabled
            and self.jepa_controller is not None
            and getattr(self.config, "route_mode", "jepa") == "jepa"
        )
        # 启发式路径：文本分类先出 plan；JEPA 路径：嵌入后再 plan_from_jepa
        plan: Optional[RoutePlan] = None
        jepa_signals: Optional[ControlSignals] = None
        if not jepa_on:
            if route_text is not None:
                plan = self.thalamus.plan_from_text(route_text)
            elif compute_vla or visual_input is not None:
                plan = self.thalamus.plan_from_text("vision planning action")
        self.last_route_plan = plan
        prev_route = self._apply_route(plan)

        try:
            text_embeds = self.embed_tokens(input_ids)
            if operator_embedding is not None:
                op = operator_embedding.to(device=device, dtype=text_embeds.dtype)
                if op.dim() == 2:
                    op = op.unsqueeze(1)
                text_embeds = text_embeds + op
            vjepa_features = None
            vjepa_loss = torch.zeros((), device=device)

            if visual_input is not None and self.vjepa is not None:
                vjepa_features, vjepa_loss, _ = self.vjepa(visual_input)

            # —— JEPA 总控：ControlSignals → 丘脑真调度 ——
            if jepa_on:
                jepa_signals = self.jepa_controller(
                    text_embeds,
                    vision_latent=vjepa_features,
                )
                plan = self.thalamus.plan_from_jepa(jepa_signals)
                # 冷启动：有 route_text 时用启发式任务类 + JEPA scale/wake
                # （未训 JEPA task_head 时仍能拉开 ticks/budget）
                if route_text:
                    heur = self.thalamus.plan_from_text(route_text)
                    plan = self.thalamus._plan(
                        heur.task_type,
                        (
                            f"jepa+text:{heur.task_type.value}:"
                            f"unc={jepa_signals.uncertainty:.2f}:"
                            f"scale={jepa_signals.compute_scale:.2f}"
                        ),
                        compute_scale=jepa_signals.compute_scale,
                        wake_override=set(plan.wake) | set(heur.wake),
                    )
                elif compute_vla or visual_input is not None:
                    heur = self.thalamus.plan_from_text("vision planning action")
                    plan = self.thalamus._plan(
                        heur.task_type,
                        f"jepa+vision:unc={jepa_signals.uncertainty:.2f}",
                        compute_scale=jepa_signals.compute_scale,
                        wake_override=set(plan.wake) | set(heur.wake),
                    )
                self._restore_route(prev_route)
                prev_route = self._apply_route(plan)
                self.last_route_plan = plan
                self.last_jepa_signals = jepa_signals
                outputs["jepa_driven"] = True
                outputs["jepa_uncertainty"] = jepa_signals.uncertainty
                outputs["jepa_compute_scale"] = jepa_signals.compute_scale
                outputs["jepa_ctrl"] = jepa_signals.jepa_ctrl
                outputs["jepa_next_latent"] = jepa_signals.next_latent
            else:
                outputs["jepa_driven"] = False

            # —— 小脑缓存命中快路径（最省算力）——
            cache_hit = None
            if plan is not None and plan.cache_lookup and route_text:
                cache_hit = self.cerebellum.lookup(route_text)
            if cache_hit is not None:
                # 命中：用缓存的 pooled hidden 直接走解码，跳过主干
                cached = cache_hit.value.to(device=device, dtype=text_embeds.dtype)
                if cached.dim() == 2:
                    cached = cached.unsqueeze(0)
                hidden_states = cached.expand(text_embeds.shape[0], -1, -1)
                outputs["cache_hit"] = True
                outputs["cache_hits"] = cache_hit.hits
            else:
                # —— 颞叶 RAG 召回（knowledge/code）——
                rag_prefix = None
                if plan is not None and plan.enable_rag and self.rag is not None and route_text:
                    hits = self.rag.search(route_text, k=2)
                    if hits:
                        # 把检索文本哈希成伪嵌入拼前缀（占位；真实走 embed）
                        vs = self.config.vocab_size
                        rag_ids = []
                        for doc, _ in hits:
                            for c in (doc.text or "")[:8]:
                                rag_ids.append((ord(c) * 131 + 7) % vs)
                        if rag_ids:
                            rag_ids_t = torch.tensor([rag_ids], dtype=torch.long, device=device)
                            rag_prefix = self.embed_tokens(rag_ids_t)
                            outputs["rag_hits"] = [d.doc_id for d, _ in hits]

                if vjepa_features is not None:
                    parts = [vjepa_features]
                    if rag_prefix is not None:
                        parts.append(rag_prefix)
                    parts.append(text_embeds)
                    hidden_states = torch.cat(parts, dim=1)
                elif rag_prefix is not None:
                    hidden_states = torch.cat([rag_prefix, text_embeds], dim=1)
                else:
                    hidden_states = text_embeds

                total_aux_loss = torch.zeros((), device=device)
                total_ctmj_loss = torch.zeros((), device=device)
                tick_list: List[int] = []
                ctm_state = None
                ctm_output = None
                inject_every = max(1, self.ctm_inject_every)

                # layer-skip 由丘脑开关决定
                dyn_skip = use_dynamic_skip or (plan is not None and plan.enable_layer_skip)

                # --- Mamba front ---
                for layer_idx in range(self.mamba2_layers):
                    if ctm_output is None or (layer_idx % inject_every == 0):
                        ctm_output, ctm_state, total_ctmj_loss = self._refresh_ctm(
                            hidden_states,
                            use_ctm_adaptive,
                            return_jepa=return_all_losses,
                            tick_list=tick_list,
                            ctmj_acc=total_ctmj_loss,
                        )

                    if dyn_skip and ctm_state is not None:
                        skip_mask = self._should_skip_layers(ctm_state, layer_idx)
                        if bool(skip_mask.all()):
                            continue

                    if self._grad_ckpt and self.training:
                        hidden_states, aux_loss, _ = torch.utils.checkpoint.checkpoint(
                            self.mamba_blocks[layer_idx],
                            hidden_states,
                            self.moe_layers[layer_idx],
                            ctm_output,
                            use_reentrant=False,
                        )
                    else:
                        hidden_states, aux_loss, _ = self.mamba_blocks[layer_idx](
                            hidden_states, self.moe_layers[layer_idx], ctm_output
                        )
                    total_aux_loss = total_aux_loss + aux_loss

                # --- MLA back ---
                for layer_idx in range(self.mla_layers):
                    global_idx = layer_idx + self.mamba2_layers
                    if ctm_output is None or (global_idx % inject_every == 0):
                        ctm_output, ctm_state, total_ctmj_loss = self._refresh_ctm(
                            hidden_states,
                            use_ctm_adaptive,
                            return_jepa=return_all_losses,
                            tick_list=tick_list,
                            ctmj_acc=total_ctmj_loss,
                        )

                    if dyn_skip and ctm_state is not None:
                        skip_mask = self._should_skip_layers(ctm_state, global_idx)
                        if bool(skip_mask.all()):
                            continue

                    if self._grad_ckpt and self.training:
                        hidden_states, aux_loss, _ = torch.utils.checkpoint.checkpoint(
                            self.mla_blocks[layer_idx],
                            hidden_states,
                            self.moe_layers[global_idx],
                            ctm_output,
                            None,
                            use_int4_cache,
                            use_reentrant=False,
                        )
                    else:
                        hidden_states, aux_loss, _ = self.mla_blocks[layer_idx](
                            hidden_states,
                            self.moe_layers[global_idx],
                            ctm_output,
                            kv_cache=None,
                            use_int4_cache=use_int4_cache,
                        )
                    total_aux_loss = total_aux_loss + aux_loss

                hidden_states = self.final_norm(hidden_states)

                # —— 海马体残差注入（reasoning/action）——
                if plan is not None and plan.enable_hippo:
                    hidden_states = hidden_states + self.hippocampus(hidden_states) * 0.1

                # —— 小脑纠错（默认开，轻量）——
                if plan is None or plan.enable_cerebellum:
                    hidden_states = self.cerebellum(hidden_states)

                # —— 顶叶校验（math/code）——
                if plan is not None and plan.enable_parietal and route_text:
                    pv = self.parietal.reason(hidden=hidden_states, text=route_text)
                    outputs["parietal_verify"] = pv  # type: ignore[assignment]

                # 写小脑缓存（仅在有 route_text 时）
                if route_text and (plan is None or plan.enable_cerebellum):
                    self.cerebellum.store(route_text, hidden_states.mean(dim=1).detach())

            # 默认 meaning_first：目标语义调制 logits（无 AR fallback）
            target_meaning = None
            recon_loss = torch.zeros((), device=device)
            jepa_ctrl = jepa_signals.jepa_ctrl if jepa_signals is not None else None
            if mode == "meaning_first":
                # 仅用文本段做意义规划（跳过视觉/RAG 前缀）
                prefix = 0
                if vjepa_features is not None:
                    prefix += vjepa_features.shape[1]
                if outputs.get("rag_hits"):
                    prefix += rag_prefix.shape[1]  # type: ignore[union-attr]
                prompt_h = hidden_states[:, prefix:, :] if prefix else hidden_states
                target_meaning = self.meaning_planner.plan(prompt_h, jepa_ctrl=jepa_ctrl)
                md_out = self.meaning_decoder(
                    hidden_states,
                    target_meaning,
                    labels=labels,
                    lm_head=self.lm_head,
                )
                logits = md_out["logits"]
                if "recon_loss" in md_out:
                    recon_loss = md_out["recon_loss"]
                outputs["target_meaning"] = target_meaning
                outputs["recon_loss"] = recon_loss
            else:
                logits = self.lm_head(hidden_states)

            outputs["logits"] = logits
            outputs["decode_mode"] = mode
            outputs["hidden_states"] = hidden_states
            if plan is not None:
                outputs["expert_budget"] = torch.tensor(plan.expert_budget, device=device)
                outputs["thalamus_task"] = plan.task_type.value
                outputs["thalamus_wake"] = sorted(plan.wake)
                outputs["thalamus_sub_regions"] = plan.sub_regions
                outputs["ctm_ticks_plan"] = plan.ctm_ticks
                outputs["thalamus_reason"] = plan.reason

            # VLA / World-Action 头（按需）
            if compute_vla and self.action_head is not None:
                vla_out = self.action_head(
                    hidden_states, proprio=proprio, action_labels=action_labels
                )
                outputs["action_logits"] = vla_out.action_logits
                outputs["action_ids"] = vla_out.action_ids
                outputs["action_names"] = vla_out.action_names  # type: ignore[assignment]
                outputs["action_mu"] = vla_out.action_mu
                if vla_out.action_loss is not None:
                    outputs["action_loss"] = vla_out.action_loss

            if compute_wa and self.world_action is not None:
                wa_out = self.world_action(
                    hidden_states,
                    next_obs_hidden=next_visual_hidden,
                    jepa_ctrl=jepa_ctrl,
                )
                outputs["world_current"] = wa_out.current_latent
                outputs["world_next"] = wa_out.next_latent
                outputs["action_prior_logits"] = wa_out.action_prior_logits
                if wa_out.world_loss is not None:
                    outputs["world_loss"] = wa_out.world_loss

            # 运动动作中枢（M1/premotor/SMA）— 仅产出 ActionSpec，执行在 serve/action/
            if plan is not None and plan.enable_action and self.motor is not None and route_text:
                motor_out = self.motor.plan_actions(
                    hidden_states, intent=route_text, jepa_ctrl=jepa_ctrl
                )
                outputs["motor_actions"] = motor_out  # type: ignore[assignment]

            avg_ticks = float(sum(tick_list) / max(1, len(tick_list))) if not outputs.get("cache_hit") else 0.0
            self.last_avg_ticks = avg_ticks
            outputs["avg_ticks"] = torch.tensor(avg_ticks, device=device)

            if return_all_losses:
                if vjepa_features is not None:
                    text_logits = logits[:, vjepa_features.shape[1] :, :]
                else:
                    text_logits = logits
                outputs["vjepa_loss"] = vjepa_loss
                outputs["ctmj_loss"] = total_ctmj_loss
                outputs["moe_loss"] = total_aux_loss
                outputs["lm_logits"] = text_logits
        finally:
            self._restore_route(prev_route)

        return outputs

    @torch.no_grad()
    def generate_meaning_first(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        route_text: Optional[str] = None,
        operator_embedding: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        """Q1B 推理：规划目标语义 → 纯意义解码；坍塌只标记不回退（Q3B）。"""
        self.eval()
        device = input_ids.device
        # 先跑一遍主干拿 prompt hidden / meaning
        out = self.forward(
            input_ids,
            route_text=route_text,
            decode_mode="meaning_first",
            operator_embedding=operator_embedding,
            return_all_losses=False,
        )
        target_meaning = out["target_meaning"]

        def _hidden_proj(emb: torch.Tensor) -> torch.Tensor:
            # 轻量投影：用 final_norm 作为逐步 hidden 近似（tiny 冒烟路径）
            return self.final_norm(emb)

        gen_ids = self.meaning_decoder.generate(
            hidden_proj=_hidden_proj,
            embed=self.embed_tokens,
            target_meaning=target_meaning,
            max_len=max_new_tokens,
            lm_head=self.lm_head,
            bos_token_id=int(getattr(self.config, "pad_token_id", 0)),
            temperature=temperature,
        )
        report = self.collapse_detector.check(gen_ids)
        return {
            "generated_ids": gen_ids,
            "target_meaning": target_meaning,
            "collapse": report,
            "route": self.last_route_plan,
            "jepa_signals": self.last_jepa_signals,
            "jepa_driven": bool(out.get("jepa_driven")),
            "jepa_uncertainty": out.get("jepa_uncertainty"),
            "decode_mode": "meaning_first",
            "ar_fallback": False,
        }

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        step: int = 0,
        total_steps: int = 100000,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        text_logits = outputs["lm_logits"]
        B, L, V = text_logits.shape
        # Align labels to logits length if needed
        if labels.shape[1] != L:
            min_l = min(labels.shape[1], L)
            text_logits = text_logits[:, :min_l, :]
            labels = labels[:, :min_l]
            B, L, V = text_logits.shape

        lm_loss = F.cross_entropy(
            text_logits.reshape(B * L, V),
            labels.reshape(B * L),
            ignore_index=-100,
        )
        recon = outputs.get("recon_loss", torch.zeros((), device=lm_loss.device))
        losses = {
            "lm": lm_loss,
            "vjepa": outputs.get(
                "vjepa_loss", torch.zeros((), device=lm_loss.device)
            ),
            "ctmj": outputs.get(
                "ctmj_loss", torch.zeros((), device=lm_loss.device)
            ),
            "moe": outputs.get(
                "moe_loss", torch.zeros((), device=lm_loss.device)
            ),
        }
        total_loss, stats = self.loss_manager.compute_loss(
            losses, self, step, total_steps
        )
        # 意义重建项（Q1B）：并入总 loss，不改变 StageAware 权重表
        recon_lambda = 0.1
        total_loss = total_loss + recon_lambda * recon
        stats["stage"] = float(self.training_stage)
        stats["lm_loss_raw"] = lm_loss.item()
        stats["recon_loss"] = float(recon.detach().item()) if torch.is_tensor(recon) else float(recon)
        stats["avg_ticks"] = float(outputs.get("avg_ticks", torch.tensor(0.0)).item())
        stats["decode_mode"] = str(outputs.get("decode_mode", self.decode_mode))
        return total_loss, stats

    def get_num_parameters(self) -> Tuple[float, float]:
        return count_parameters(self.config)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
