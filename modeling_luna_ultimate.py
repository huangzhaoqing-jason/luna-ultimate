"""Luna-Ultimate Ultimate: 550B Hybrid Architecture with V-JEPA + CTM-JEPA.

Fused architecture combining:
  - V-JEPA: External visual world model (video/image understanding)
  - Mamba2-SSD: Fast contextual encoding (Layers 1-12)
  - MLA: Deep reasoning with KV compression (Layers 13-32)
  - FlashMoE: Sparse expert activation (All 32 layers)
  - CTM-JEPA: Internal thinking state prediction (Global)
  - Stage-aware training: 3-stage progressive training

Forward pass flow:
  Text Input → [Embedding] ─────────────────────────────┐
  Visual Input → [V-JEPA Encoder] → [Projector] ────────┤
                                                          ├→ [Mamba2×12 + MoE] ─→ [MLA×20 + MoE] ─→ [LM Head]
  CTM ←───────────────────────────────────────────────── (injected into every layer)
  CTM-JEPA ←──────────────────────────────────────────── (predicts future thinking states)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict

from config import LunaConfig, count_parameters, verify_parameters
from modeling_ctm import CTM
from modeling_vjepa import VJEPA
from modeling_mamba2 import Mamba2Block
from modeling_mla import MLABlock
from modeling_flashmoe import FlashMoE
from quant_utils import RMSNorm
from loss_manager import StageAwareLossManager


class LunaUltimateFused(nn.Module):
    """Luna-Ultimate: 550B Fused V-JEPA + CTM-JEPA + Hybrid Architecture.

    Architecture flow:
      Text: [B, L_txt] → Embedding → [B, L_txt, d_model]
      Visual: [B, C, T, H, W] → V-JEPA → [B, N_vis, embed_dim] → Projector → [B, N_vis, d_model]
      Concat: [B, L_txt+N_vis, d_model]
      → Mamba2-SSD (L1-12) + CTM residual + FlashMoE
      → MLA (L13-32) + CTM residual + FlashMoE
      → Final Norm → LM Head → [B, L, vocab_size]

    Args:
        config: LunaConfig with all hyperparameters.
    """

    def __init__(self, config: LunaConfig):
        super().__init__()
        self.config = config
        self.d_model = config.hidden_size
        self.vocab_size = config.vocab_size
        self.num_layers = config.num_hidden_layers
        self.mamba2_layers = config.mamba2_layers
        self.mla_layers = config.mla_layers

        # ==================== Text Embedding ====================
        self.embed_tokens = nn.Embedding(config.vocab_size, self.d_model)

        # ==================== V-JEPA (Visual Encoder) ====================
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

        # ==================== Global CTM (with JEPA) ====================
        self.ctm = CTM(config)

        # ==================== Mamba2 blocks (Layers 1-12) ====================
        self.mamba_blocks = nn.ModuleList([
            Mamba2Block(config, layer_idx=i)
            for i in range(self.mamba2_layers)
        ])

        # ==================== MLA blocks (Layers 13-32) ====================
        self.mla_blocks = nn.ModuleList([
            MLABlock(config, layer_idx=i + self.mamba2_layers)
            for i in range(self.mla_layers)
        ])

        # ==================== FlashMoE layers (one per layer, 32 total) ====================
        self.moe_layers = nn.ModuleList([
            FlashMoE(config) for _ in range(self.num_layers)
        ])

        # ==================== Final norm + LM Head ====================
        self.final_norm = RMSNorm(self.d_model, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(self.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight  # Weight tying

        # ==================== Layer skip probes ====================
        self.skip_probe_12 = nn.Linear(config.ctm_n_neurons, 1, bias=False)
        self.skip_probe_24 = nn.Linear(config.ctm_n_neurons, 1, bias=False)

        # ==================== Loss Manager ====================
        self.loss_manager = StageAwareLossManager(stage=1, use_uncertainty=True)

        # ==================== Training stage tracking ====================
        self.training_stage = 1

        # Print verification
        self._verify_and_print_params()

    def _verify_and_print_params(self):
        verify_parameters(self.config)

    def set_training_stage(self, stage: int):
        """Set training stage (1, 2, or 3) and freeze/unfreeze modules."""
        self.training_stage = stage
        self.loss_manager.set_stage(stage)

        # Freeze/unfreeze modules based on stage
        freeze_info = self.loss_manager.get_frozen_modules()
        if stage in freeze_info:
            freeze_pattern = freeze_info[stage]
            if freeze_pattern == "vjepa":
                if stage == 1:
                    # Stage 1: freeze everything except V-JEPA
                    for name, param in self.named_parameters():
                        if "vjepa" not in name:
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
                elif stage == 2:
                    # Stage 2: freeze V-JEPA, train rest
                    for name, param in self.named_parameters():
                        if "vjepa" in name:
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
            else:
                # Stage 3: train everything
                for param in self.parameters():
                    param.requires_grad = True

    def _should_skip_layers(
        self, ctm_state: torch.Tensor, layer_idx: int
    ) -> torch.Tensor:
        """Determine which tokens can skip layers."""
        if layer_idx == 12:
            probe = self.skip_probe_12(ctm_state)
        elif layer_idx == 24:
            probe = self.skip_probe_24(ctm_state)
        else:
            return torch.zeros(ctm_state.shape[0], dtype=torch.bool, device=ctm_state.device)
        skip_prob = torch.sigmoid(probe.squeeze(-1))
        return skip_prob > self.config.layer_skip_prob_threshold

    def forward(
        self,
        input_ids: torch.Tensor,
        visual_input: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_ctm_adaptive: bool = True,
        use_dynamic_skip: bool = False,
        return_all_losses: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass of Luna-Ultimate Fused.

        Args:
            input_ids: [B, L_txt] — text token indices.
            visual_input: Optional [B, C, T, H, W] — visual input.
            attention_mask: Optional [B, L] — padding mask.
            use_ctm_adaptive: Whether to use CTM adaptive early exit.
            use_dynamic_skip: Whether to use dynamic layer skipping.
            return_all_losses: Return all loss components (for training).

        Returns:
            Dict with keys:
              - "logits": [B, L, vocab_size]
              - "lm_loss": Next-token prediction loss
              - "vjepa_loss": V-JEPA prediction loss
              - "ctmj_loss": CTM-JEPA prediction loss
              - "moe_loss": MoE load balancing loss
              - "total_loss": Combined loss (if return_all_losses=True)
        """
        B, L_txt = input_ids.shape
        device = input_ids.device
        outputs = {}

        # ==================== Text Embedding ====================
        text_embeds = self.embed_tokens(input_ids)  # [B, L_txt, d_model]

        # ==================== V-JEPA Visual Encoding ====================
        vjepa_features = None
        vjepa_loss = torch.tensor(0.0, device=device)

        if visual_input is not None:
            vjepa_features, vjepa_loss, _ = self.vjepa(visual_input)
            # vjepa_features: [B, N_vis, d_model=8192]

        # ==================== Concatenate Text + Visual ====================
        if vjepa_features is not None:
            hidden_states = torch.cat([vjepa_features, text_embeds], dim=1)
            total_len = hidden_states.shape[1]
        else:
            hidden_states = text_embeds
            total_len = L_txt

        # ==================== Main Forward Pass ====================
        total_aux_loss = torch.tensor(0.0, device=device)
        total_ctmj_loss = torch.tensor(0.0, device=device)
        ctm_state = None

        # --- Layers 1-12: Mamba2-SSD + FlashMoE ---
        for layer_idx in range(self.mamba2_layers):
            if use_dynamic_skip and ctm_state is not None:
                skip_mask = self._should_skip_layers(ctm_state, layer_idx)
                if skip_mask.any():
                    processed = hidden_states[~skip_mask]
                    if processed.shape[0] > 0:
                        ctm_output, _, _, ctmj = self.ctm(
                            processed, use_adaptive_early_exit=use_ctm_adaptive,
                            return_jepa_loss=True,
                        )
                        processed, aux_loss, _ = self.mamba_blocks[layer_idx](
                            processed, self.moe_layers[layer_idx], ctm_output
                        )
                        hidden_states[~skip_mask] = processed
                        total_aux_loss = total_aux_loss + aux_loss
                        if ctmj is not None:
                            total_ctmj_loss = total_ctmj_loss + ctmj
                    continue

            ctm_output, _, _, ctmj = self.ctm(
                hidden_states, use_adaptive_early_exit=use_ctm_adaptive,
                return_jepa_loss=True,
            )
            ctm_state = self.ctm.get_ctm_state(hidden_states)

            hidden_states, aux_loss, _ = self.mamba_blocks[layer_idx](
                hidden_states, self.moe_layers[layer_idx], ctm_output
            )
            total_aux_loss = total_aux_loss + aux_loss
            if ctmj is not None:
                total_ctmj_loss = total_ctmj_loss + ctmj

        # --- Layers 13-32: MLA + FlashMoE ---
        for layer_idx in range(self.mla_layers):
            global_idx = layer_idx + self.mamba2_layers

            if use_dynamic_skip and ctm_state is not None:
                skip_mask = self._should_skip_layers(ctm_state, global_idx)
                if skip_mask.any():
                    processed = hidden_states[~skip_mask]
                    if processed.shape[0] > 0:
                        ctm_output, _, _, ctmj = self.ctm(
                            processed, use_adaptive_early_exit=use_ctm_adaptive,
                            return_jepa_loss=True,
                        )
                        processed, aux_loss, _ = self.mla_blocks[layer_idx](
                            processed, self.moe_layers[global_idx], ctm_output
                        )
                        hidden_states[~skip_mask] = processed
                        total_aux_loss = total_aux_loss + aux_loss
                        if ctmj is not None:
                            total_ctmj_loss = total_ctmj_loss + ctmj
                    continue

            ctm_output, _, _, ctmj = self.ctm(
                hidden_states, use_adaptive_early_exit=use_ctm_adaptive,
                return_jepa_loss=True,
            )
            ctm_state = self.ctm.get_ctm_state(hidden_states)

            hidden_states, aux_loss, _ = self.mla_blocks[layer_idx](
                hidden_states, self.moe_layers[global_idx], ctm_output
            )
            total_aux_loss = total_aux_loss + aux_loss
            if ctmj is not None:
                total_ctmj_loss = total_ctmj_loss + ctmj

        # ==================== Final Norm + LM Head ====================
        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)  # [B, total_len, vocab_size]

        outputs["logits"] = logits

        # ==================== Loss Computation ====================
        if return_all_losses:
            # Extract text-only logits for LM loss
            if vjepa_features is not None:
                text_logits = logits[:, vjepa_features.shape[1]:, :]
            else:
                text_logits = logits

            outputs["vjepa_loss"] = vjepa_loss
            outputs["ctmj_loss"] = total_ctmj_loss
            outputs["moe_loss"] = total_aux_loss

            # LM loss computed externally (needs labels)
            outputs["lm_logits"] = text_logits

        return outputs

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        step: int = 0,
        total_steps: int = 100000,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute all losses and return weighted total.

        Args:
            outputs: Dict from forward().
            labels: [B, L_txt] — target token IDs.
            step: Current training step.
            total_steps: Total training steps.

        Returns:
            total_loss: scalar.
            stats: dict of all loss components.
        """
        # Next-token prediction loss
        text_logits = outputs["lm_logits"]
        B, L, V = text_logits.shape
        lm_loss = F.cross_entropy(
            text_logits.view(B * L, V),
            labels.view(B * L),
            ignore_index=-100,
        )

        # Collect all losses
        losses = {
            "lm": lm_loss,
            "vjepa": outputs.get("vjepa_loss", torch.tensor(0.0, device=lm_loss.device)),
            "ctmj": outputs.get("ctmj_loss", torch.tensor(0.0, device=lm_loss.device)),
            "moe": outputs.get("moe_loss", torch.tensor(0.0, device=lm_loss.device)),
        }

        total_loss, stats = self.loss_manager.compute_loss(losses, self, step, total_steps)
        stats["stage"] = float(self.training_stage)
        stats["lm_loss_raw"] = lm_loss.item()

        return total_loss, stats

    def get_num_parameters(self) -> Tuple[float, float]:
        return count_parameters(self.config)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device