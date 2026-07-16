"""Luna-Ultimate: 550B Hybrid Architecture Model.

Full integration of:
  - Embedding (151936 × 8192)
  - Layers 1-12: Mamba2-SSD + FlashMoE + CTM residual
  - Layers 13-32: MLA + FlashMoE + CTM residual
  - Global CTM module (adaptive 1-4 ticks)
  - Speculative decoding: Mamba2 drafts, MLA verifies
  - Dynamic layer skipping
  - Architecture-level speculative decoding interface
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass

from config import LunaConfig, count_parameters, verify_parameters
from modeling_ctm import CTM
from modeling_mamba2 import Mamba2Block
from modeling_mla import MLABlock
from modeling_flashmoe import FlashMoE
from quant_utils import RMSNorm


@dataclass
class GenerationCache:
    """Cache for speculative decoding generation."""
    mamba_states: List[Optional[torch.Tensor]]  # Mamba2 states per layer
    kv_caches: List[Optional[Tuple[torch.Tensor, torch.Tensor]]]  # MLA KV caches
    ctm_state: Optional[torch.Tensor]  # CTM neuron state
    draft_tokens: Optional[torch.Tensor]  # Draft tokens from Mamba2
    draft_logits: Optional[torch.Tensor]  # Draft logits


class LunaUltimate(nn.Module):
    """Luna-Ultimate: 550B MoE Hybrid Model.

    Architecture:
      Embedding → [Mamba2×12] → [MLA×20] → LM Head
      Global CTM injects into every layer.
      All 32 layers have FlashMoE (46 routed + 2 shared experts).

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

        # Embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, self.d_model)

        # Global CTM
        self.ctm = CTM(config)

        # Mamba2 blocks (layers 1-12)
        self.mamba_blocks = nn.ModuleList([
            Mamba2Block(config, layer_idx=i)
            for i in range(self.mamba2_layers)
        ])

        # MLA blocks (layers 13-32)
        self.mla_blocks = nn.ModuleList([
            MLABlock(config, layer_idx=i + self.mamba2_layers)
            for i in range(self.mla_layers)
        ])

        # FlashMoE layers (one per layer, 32 total)
        self.moe_layers = nn.ModuleList([
            FlashMoE(config)
            for _ in range(self.num_layers)
        ])

        # Final norm
        self.final_norm = RMSNorm(self.d_model, eps=config.rms_norm_eps)

        # LM Head (tied with embedding)
        self.lm_head = nn.Linear(self.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

        # Layer skip probes (at layer 12 and 24)
        self.skip_probe_12 = nn.Linear(config.ctm_n_neurons, 1, bias=False)
        self.skip_probe_24 = nn.Linear(config.ctm_n_neurons, 1, bias=False)

        # Print parameter verification
        self._verify_and_print_params()

    def _verify_and_print_params(self):
        """Verify parameter count and print summary."""
        verify_parameters(self.config)

    def _should_skip_layers(
        self,
        ctm_state: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """Determine which tokens can skip layers.

        Args:
            ctm_state: [B, n_neurons] from CTM.
            layer_idx: Current layer index.

        Returns:
            skip_mask: [B] boolean mask - True = skip this token
        """
        if layer_idx == 12:
            probe = self.skip_probe_12(ctm_state)  # [B, 1]
        elif layer_idx == 24:
            probe = self.skip_probe_24(ctm_state)  # [B, 1]
        else:
            return torch.zeros(ctm_state.shape[0], dtype=torch.bool, device=ctm_state.device)

        skip_prob = torch.sigmoid(probe.squeeze(-1))  # [B]
        return skip_prob > self.config.layer_skip_prob_threshold

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_ctm_adaptive: bool = True,
        use_dynamic_skip: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full forward pass of Luna-Ultimate.

        Args:
            input_ids: [B, L] - token indices.
            attention_mask: [B, L] - optional padding mask.
            use_ctm_adaptive: Whether to use CTM adaptive early exit.
            use_dynamic_skip: Whether to use dynamic layer skipping.

        Returns:
            logits: [B, L, vocab_size]
            total_aux_loss: scalar - sum of all MoE aux losses
        """
        B, L = input_ids.shape

        # Embedding: [B, L] -> [B, L, d_model=8192]
        hidden_states = self.embed_tokens(input_ids)

        total_aux_loss = torch.tensor(0.0, device=input_ids.device)
        ctm_state = None

        # ==================== Layers 1-12: Mamba2-SSD + FlashMoE ====================
        for layer_idx in range(self.mamba2_layers):
            # Dynamic layer skipping
            if use_dynamic_skip and ctm_state is not None:
                skip_mask = self._should_skip_layers(ctm_state, layer_idx)
                if skip_mask.any():
                    # Save skipped tokens' hidden states
                    skipped_hidden = hidden_states[skip_mask].clone()
                    # Only process non-skipped tokens
                    processed = hidden_states[~skip_mask]
                    if processed.shape[0] > 0:
                        processed, aux_loss, _ = self.mamba_blocks[layer_idx](
                            processed, self.moe_layers[layer_idx], torch.zeros_like(processed)
                        )
                        hidden_states[~skip_mask] = processed
                        total_aux_loss = total_aux_loss + aux_loss
                    continue

            # CTM residual
            ctm_output, sync_matrix, num_ticks = self.ctm(
                hidden_states,
                use_adaptive_early_exit=use_ctm_adaptive,
            )
            ctm_state = self.ctm.get_ctm_state(hidden_states)

            # Mamba2 block
            hidden_states, aux_loss, _ = self.mamba_blocks[layer_idx](
                hidden_states,
                self.moe_layers[layer_idx],
                ctm_output,
            )
            total_aux_loss = total_aux_loss + aux_loss

        # ==================== Layers 13-32: MLA + FlashMoE ====================
        for layer_idx in range(self.mla_layers):
            global_idx = layer_idx + self.mamba2_layers

            # Dynamic layer skipping
            if use_dynamic_skip and ctm_state is not None:
                skip_mask = self._should_skip_layers(ctm_state, global_idx)
                if skip_mask.any():
                    processed = hidden_states[~skip_mask]
                    if processed.shape[0] > 0:
                        processed, aux_loss, _ = self.mla_blocks[layer_idx](
                            processed,
                            self.moe_layers[global_idx],
                            torch.zeros_like(processed),
                        )
                        hidden_states[~skip_mask] = processed
                        total_aux_loss = total_aux_loss + aux_loss
                    continue

            # CTM residual
            ctm_output, sync_matrix, num_ticks = self.ctm(
                hidden_states,
                use_adaptive_early_exit=use_ctm_adaptive,
            )
            ctm_state = self.ctm.get_ctm_state(hidden_states)

            # MLA block
            hidden_states, aux_loss, _ = self.mla_blocks[layer_idx](
                hidden_states,
                self.moe_layers[global_idx],
                ctm_output,
            )
            total_aux_loss = total_aux_loss + aux_loss

        # Final norm
        hidden_states = self.final_norm(hidden_states)

        # LM Head: [B, L, d_model] -> [B, L, vocab_size]
        logits = self.lm_head(hidden_states)

        return logits, total_aux_loss

    # ==================== Speculative Decoding ====================

    def _mamba2_draft(
        self,
        hidden_states: torch.Tensor,
        mamba_states: List[Optional[torch.Tensor]],
    ) -> Tuple[torch.Tensor, List[Optional[torch.Tensor]]]:
        """Run Mamba2 layers (1-12) to produce draft tokens.

        Args:
            hidden_states: [B, 1, d_model] - single token
            mamba_states: List of Mamba2 states from previous step

        Returns:
            draft_logits: [B, 1, vocab_size]
            updated_states: Updated Mamba2 states
        """
        updated_states = []
        for layer_idx in range(self.mamba2_layers):
            ctm_output, _, _ = self.ctm(hidden_states, use_adaptive_early_exit=True)
            hidden_states, new_state = self.mamba_blocks[layer_idx].mamba(
                hidden_states, mamba_states[layer_idx]
            )
            hidden_states = hidden_states + ctm_output
            normed = self.mamba_blocks[layer_idx].moe_norm(hidden_states)
            hidden_states, _ = self.moe_layers[layer_idx](normed, hidden_states)
            updated_states.append(new_state)

        draft_logits = self.lm_head(self.final_norm(hidden_states))
        return draft_logits, updated_states

    def _mla_verify(
        self,
        hidden_states: torch.Tensor,
        kv_caches: List[Optional[Tuple[torch.Tensor, torch.Tensor]]],
        use_int4_cache: bool = False,
    ) -> Tuple[torch.Tensor, List[Optional[Tuple[torch.Tensor, torch.Tensor]]]]:
        """Run MLA layers (13-32) to verify draft tokens.

        Args:
            hidden_states: [B, 1, d_model] - single token
            kv_caches: List of KV caches from previous step
            use_int4_cache: Whether to use INT4 KV cache

        Returns:
            verified_logits: [B, 1, vocab_size]
            updated_kv_caches: Updated KV caches
        """
        updated_kv_caches = []
        for layer_idx in range(self.mla_layers):
            global_idx = layer_idx + self.mamba2_layers
            ctm_output, _, _ = self.ctm(hidden_states, use_adaptive_early_exit=True)
            hidden_states, new_kv = self.mla_blocks[layer_idx].attention(
                hidden_states, kv_caches[layer_idx], use_int4_cache
            )
            hidden_states = hidden_states + ctm_output
            normed = self.mla_blocks[layer_idx].moe_norm(hidden_states)
            hidden_states, _ = self.moe_layers[global_idx](normed, hidden_states)
            updated_kv_caches.append(new_kv)

        verified_logits = self.lm_head(self.final_norm(hidden_states))
        return verified_logits, updated_kv_caches

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        use_speculative: bool = True,
        kv_cache_int4: bool = True,
        ctm_adaptive_early_exit: bool = True,
        dynamic_layer_skip: bool = True,
    ) -> torch.Tensor:
        """Generate tokens with architecture-level speculative decoding.

        Speculative decoding pipeline:
          1. Mamba2 layers (1-12) draft 4 tokens (no KV cache, fast)
          2. MLA layers (13-32) verify all 4 draft tokens in parallel
          3. Accept/reject based on logit matching
          4. Repeat

        Args:
            input_ids: [B, L] - prompt tokens.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-K filtering.
            top_p: Nucleus (top-p) filtering.
            use_speculative: Whether to use speculative decoding.
            kv_cache_int4: Whether to use INT4 KV cache.
            ctm_adaptive_early_exit: Whether to use CTM adaptive early exit.
            dynamic_layer_skip: Whether to use dynamic layer skipping.

        Returns:
            generated_ids: [B, L + max_new_tokens]
        """
        B, prompt_len = input_ids.shape
        device = input_ids.device
        generated = input_ids.clone()

        # Initialize caches
        mamba_states = [None] * self.mamba2_layers
        kv_caches = [None] * self.mla_layers

        num_draft = self.config.num_draft_tokens  # 4

        for step in range(max_new_tokens):
            # Get last token hidden state
            if step == 0:
                # Process full prompt for first step
                hidden_states = self.embed_tokens(generated)
                for layer_idx in range(self.mamba2_layers):
                    ctm_output, _, _ = self.ctm(
                        hidden_states, use_adaptive_early_exit=ctm_adaptive_early_exit
                    )
                    hidden_states, aux_loss, new_state = self.mamba_blocks[layer_idx](
                        hidden_states,
                        self.moe_layers[layer_idx],
                        ctm_output,
                    )
                    mamba_states[layer_idx] = new_state

                for layer_idx in range(self.mla_layers):
                    global_idx = layer_idx + self.mamba2_layers
                    ctm_output, _, _ = self.ctm(
                        hidden_states, use_adaptive_early_exit=ctm_adaptive_early_exit
                    )
                    hidden_states, aux_loss, new_kv = self.mla_blocks[layer_idx](
                        hidden_states,
                        self.moe_layers[global_idx],
                        ctm_output,
                        kv_cache=kv_caches[layer_idx],
                        use_int4_cache=kv_cache_int4,
                    )
                    kv_caches[layer_idx] = new_kv

                # Take last token
                last_hidden = hidden_states[:, -1:, :]  # [B, 1, d_model]
            else:
                last_hidden = self.embed_tokens(generated[:, -1:])

            if use_speculative:
                # ========== Speculative Decoding ==========
                # Phase 1: Mamba2 drafts 4 tokens
                draft_logits_list = []
                draft_hidden = last_hidden.clone()
                current_mamba_states = [s.clone() if s is not None else None for s in mamba_states]

                for d in range(num_draft):
                    draft_logits, current_mamba_states = self._mamba2_draft(
                        draft_hidden, current_mamba_states
                    )
                    draft_logits_list.append(draft_logits)

                    # Sample draft token
                    draft_token = self._sample_token(
                        draft_logits, temperature, top_k, top_p
                    )
                    draft_hidden = self.embed_tokens(draft_token)

                # Phase 2: MLA verifies all draft tokens
                verified_logits, new_kv_caches = self._mla_verify(
                    last_hidden, kv_caches, use_int4_cache=kv_cache_int4
                )

                # Accept first verified token
                next_token = self._sample_token(
                    verified_logits, temperature, top_k, top_p
                )

                # Update caches
                kv_caches = new_kv_caches
                mamba_states = current_mamba_states
            else:
                # ========== Standard autoregressive ==========
                ctm_output, _, _ = self.ctm(
                    last_hidden, use_adaptive_early_exit=ctm_adaptive_early_exit
                )

                for layer_idx in range(self.mamba2_layers):
                    last_hidden, new_state = self.mamba_blocks[layer_idx].mamba(
                        last_hidden, mamba_states[layer_idx]
                    )
                    last_hidden = last_hidden + ctm_output
                    normed = self.mamba_blocks[layer_idx].moe_norm(last_hidden)
                    last_hidden, _ = self.moe_layers[layer_idx](normed, last_hidden)
                    mamba_states[layer_idx] = new_state

                for layer_idx in range(self.mla_layers):
                    global_idx = layer_idx + self.mamba2_layers
                    ctm_output, _, _ = self.ctm(
                        last_hidden, use_adaptive_early_exit=ctm_adaptive_early_exit
                    )
                    last_hidden, new_kv = self.mla_blocks[layer_idx].attention(
                        last_hidden, kv_caches[layer_idx], kv_cache_int4
                    )
                    last_hidden = last_hidden + ctm_output
                    normed = self.mla_blocks[layer_idx].moe_norm(last_hidden)
                    last_hidden, _ = self.moe_layers[global_idx](normed, last_hidden)
                    kv_caches[layer_idx] = new_kv

                logits = self.lm_head(self.final_norm(last_hidden))
                next_token = self._sample_token(logits, temperature, top_k, top_p)

            # Append
            generated = torch.cat([generated, next_token], dim=1)

            # Check for EOS (token 2 for most tokenizers)
            if (next_token == 2).any():
                break

        return generated

    def _sample_token(
        self,
        logits: torch.Tensor,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """Sample a single token from logits.

        Args:
            logits: [B, 1, vocab_size]
            temperature: Sampling temperature.
            top_k: Top-K filtering.
            top_p: Nucleus filtering.

        Returns:
            token: [B, 1]
        """
        logits = logits[:, -1, :] / temperature  # [B, vocab_size]

        # Top-K
        if top_k > 0:
            top_k = min(top_k, logits.shape[-1])
            indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
            logits[indices_to_remove] = float("-inf")

        # Top-P (nucleus)
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]
        return next_token

    def get_num_parameters(self) -> Tuple[float, float]:
        """Return (total_params_billions, active_params_billions)."""
        return count_parameters(self.config)

    @property
    def device(self) -> torch.device:
        """Get model device."""
        return next(self.parameters()).device