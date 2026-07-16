"""Luna-Ultimate Fused: hybrid CTM × Mamba2 × MLA × FlashMoE × JEPA."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import LunaConfig, count_parameters, verify_parameters
from loss_manager import StageAwareLossManager
from modeling_ctm import CTM
from modeling_flashmoe import FlashMoE
from modeling_mamba2 import Mamba2Block
from modeling_mla import MLABlock
from quant_utils import RMSNorm


class LunaUltimateFused(nn.Module):
    """Fused Luna model with block-level CTM injection."""

    def __init__(self, config: LunaConfig):
        super().__init__()
        self.config = config
        self.d_model = config.hidden_size
        self.vocab_size = config.vocab_size
        self.num_layers = config.num_hidden_layers
        self.mamba2_layers = config.mamba2_layers
        self.mla_layers = config.mla_layers
        self.ctm_inject_every = getattr(config, "ctm_inject_every", 4)
        self._grad_ckpt = False

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

    def forward(
        self,
        input_ids: torch.Tensor,
        visual_input: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_ctm_adaptive: bool = True,
        use_dynamic_skip: bool = False,
        return_all_losses: bool = False,
        use_int4_cache: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        del attention_mask  # reserved
        B, L_txt = input_ids.shape
        device = input_ids.device
        outputs: Dict[str, torch.Tensor] = {}
        if use_int4_cache is None:
            use_int4_cache = bool(self.config.use_kv_cache_int4) and not self.training

        text_embeds = self.embed_tokens(input_ids)
        vjepa_features = None
        vjepa_loss = torch.zeros((), device=device)

        if visual_input is not None and self.vjepa is not None:
            vjepa_features, vjepa_loss, _ = self.vjepa(visual_input)

        if vjepa_features is not None:
            hidden_states = torch.cat([vjepa_features, text_embeds], dim=1)
        else:
            hidden_states = text_embeds

        total_aux_loss = torch.zeros((), device=device)
        total_ctmj_loss = torch.zeros((), device=device)
        tick_list: List[int] = []
        ctm_state = None
        ctm_output = None
        inject_every = max(1, self.ctm_inject_every)

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

            if use_dynamic_skip and ctm_state is not None:
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

            if use_dynamic_skip and ctm_state is not None:
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
        logits = self.lm_head(hidden_states)
        outputs["logits"] = logits

        avg_ticks = float(sum(tick_list) / max(1, len(tick_list)))
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

        return outputs

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
        stats["stage"] = float(self.training_stage)
        stats["lm_loss_raw"] = lm_loss.item()
        stats["avg_ticks"] = float(outputs.get("avg_ticks", torch.tensor(0.0)).item())
        return total_loss, stats

    def get_num_parameters(self) -> Tuple[float, float]:
        return count_parameters(self.config)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
