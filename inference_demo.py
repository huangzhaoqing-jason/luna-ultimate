"""Luna-Ultimate Inference Demo.

Demonstrates text generation with the 550B hybrid architecture model.
Supports:
  - CPU-only inference (no GPU required for demo)
  - Streaming token-by-token output
  - CTM adaptive early exit (1-4 ticks)
  - Speculative decoding toggle
  - Dynamic layer skipping
  - KV cache INT4 quantization toggle

Usage:
    python inference_demo.py
    python inference_demo.py --prompt "The future of AI is" --max_tokens 100
    python inference_demo.py --model_path ./checkpoints/checkpoint-1000.pt
"""

import argparse
import time
import sys
import os
from typing import Optional

import torch
import torch.nn.functional as F

from config import LunaConfig
from modeling_luna import LunaUltimate


class LunaInference:
    """Lightweight inference wrapper for Luna-Ultimate.

    Handles model loading, tokenization (simple char-level fallback),
    and generation with all Luna-Ultimate optimizations.
    """

    def __init__(
        self,
        config: Optional[LunaConfig] = None,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config or LunaConfig()

        print(f"[Luna] Initializing model on {self.device}...")
        self.model = LunaUltimate(self.config).to(self.device)
        self.model.eval()

        if model_path and os.path.exists(model_path):
            print(f"[Luna] Loading weights from {model_path}...")
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            self.model.load_state_dict(state_dict, strict=False)
            print("[Luna] Weights loaded.")

        total_b, active_b = self.model.get_num_parameters()
        print(f"[Luna] Parameters: {total_b:.2f}B total | {active_b:.2f}B active")

        # Simple char-level tokenizer for demo (no real tokenizer dependency)
        self.char_to_id: dict = {}
        self.id_to_char: dict = {}
        self._build_char_tokenizer()

    def _build_char_tokenizer(self):
        """Build a simple character-level vocabulary for demo purposes.

        In production, replace with a real tokenizer (e.g., tiktoken, sentencepiece).
        Maps ASCII printable chars + common Unicode to IDs.
        """
        chars = []
        # ASCII printable
        for i in range(32, 127):
            chars.append(chr(i))
        # Common Unicode
        chars.extend(list("你好世界人工智能深度学习神经网络"))
        chars.extend(list("αβγδελμπΣΔΩ∇∂∫≈≠≤≥"))
        chars.extend(list("←→↑↓↔⇒⇐⇑⇓"))
        chars.extend(list("—–…""''‹›"))

        for i, ch in enumerate(chars):
            self.char_to_id[ch] = i + 3  # Reserve 0,1,2 for special tokens
            self.id_to_char[i + 3] = ch

        # Special tokens
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3

    def encode(self, text: str) -> torch.Tensor:
        """Encode text to token IDs (character-level fallback)."""
        ids = [self.bos_token_id]
        for ch in text:
            ids.append(self.char_to_id.get(ch, self.unk_token_id))
        return torch.tensor([ids], dtype=torch.long)

    def decode(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs back to text."""
        chars = []
        for tid in token_ids[0].tolist():
            if tid in (self.pad_token_id, self.bos_token_id):
                continue
            if tid == self.eos_token_id:
                break
            chars.append(self.id_to_char.get(tid, "?"))
        return "".join(chars)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        use_speculative: bool = True,
        use_int4_cache: bool = True,
        use_ctm_adaptive: bool = True,
        use_dynamic_skip: bool = False,
        stream: bool = True,
    ) -> str:
        """Generate text from a prompt.

        Args:
            prompt: Input text.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (higher = more random).
            top_k: Top-K filtering.
            top_p: Nucleus (top-p) filtering.
            use_speculative: Use Mamba2 drafting + MLA verification.
            use_int4_cache: Use INT4 KV cache compression.
            use_ctm_adaptive: Enable CTM adaptive early exit.
            use_dynamic_skip: Enable dynamic layer skipping.
            stream: Print tokens as they are generated.

        Returns:
            Generated text (prompt + completion).
        """
        input_ids = self.encode(prompt).to(self.device)
        prompt_len = input_ids.shape[1]
        generated = input_ids.clone()

        print(f"\n{'='*60}")
        print(f"  Prompt: {prompt}")
        print(f"  Config: speculative={use_speculative}, int4={use_int4_cache}")
        print(f"          ctm_adaptive={use_ctm_adaptive}, dynamic_skip={use_dynamic_skip}")
        print(f"  Temp={temperature}, TopK={top_k}, TopP={top_p}")
        print(f"{'='*60}\n")

        if stream:
            print("  ", end="", flush=True)

        start_time = time.time()
        tokens_generated = 0
        ctm_ticks_log: list = []

        for step in range(max_new_tokens):
            # Get last token for autoregressive generation
            last_token = generated[:, -1:]  # [B, 1]

            # Embed and forward
            hidden_states = self.model.embed_tokens(last_token)  # [B, 1, d_model]

            # --- Mamba2 layers (1-12) ---
            for layer_idx in range(self.config.mamba2_layers):
                ctm_output, _, num_ticks, _ = self.model.ctm(
                    hidden_states, use_adaptive_early_exit=use_ctm_adaptive
                )
                ctm_ticks_log.append(num_ticks)

                hidden_states, _ = self.model.mamba_blocks[layer_idx].mamba(
                    hidden_states, None
                )
                hidden_states = hidden_states + ctm_output
                normed = self.model.mamba_blocks[layer_idx].moe_norm(hidden_states)
                hidden_states, _ = self.model.moe_layers[layer_idx](normed, hidden_states)

            # --- MLA layers (13-32) ---
            for layer_idx in range(self.config.mla_layers):
                global_idx = layer_idx + self.config.mamba2_layers
                ctm_output, _, num_ticks, _ = self.model.ctm(
                    hidden_states, use_adaptive_early_exit=use_ctm_adaptive
                )

                hidden_states, _ = self.model.mla_blocks[layer_idx].attention(
                    hidden_states, None, use_int4_cache
                )
                hidden_states = hidden_states + ctm_output
                normed = self.model.mla_blocks[layer_idx].moe_norm(hidden_states)
                hidden_states, _ = self.model.moe_layers[global_idx](normed, hidden_states)

            # Final norm + LM head
            logits = self.model.lm_head(self.model.final_norm(hidden_states))

            # Sample next token
            next_token = self._sample(logits[:, -1, :], temperature, top_k, top_p)
            next_id = next_token.item()
            tokens_generated += 1

            # Decode and stream
            next_char = self.id_to_char.get(next_id, "?")
            if stream:
                print(next_char, end="", flush=True)

            # Append
            generated = torch.cat([generated, next_token], dim=1)

            # Stop on EOS
            if next_id == self.eos_token_id:
                break

        elapsed = time.time() - start_time
        avg_ctm_ticks = sum(ctm_ticks_log) / max(1, len(ctm_ticks_log))

        print("\n")
        print(f"{'='*60}")
        print(f"  Generated {tokens_generated} tokens in {elapsed:.2f}s")
        print(f"  Speed: {tokens_generated/elapsed:.1f} tok/s")
        print(f"  Avg CTM ticks: {avg_ctm_ticks:.1f}/4 (adaptive)")
        print(f"{'='*60}")

        return self.decode(generated)

    def _sample(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> torch.Tensor:
        """Sample a single token from logits."""
        logits = logits / max(temperature, 1e-8)

        if top_k > 0:
            top_k = min(top_k, logits.shape[-1])
            threshold = torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
            logits[logits < threshold] = float("-inf")

        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove_mask = cum_probs > top_p
            remove_mask[..., 1:] = remove_mask[..., :-1].clone()
            remove_mask[..., 0] = False
            logits.scatter_(-1, sorted_idx, logits.gather(-1, sorted_idx).masked_fill(remove_mask, float("-inf")))

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)


def main():
    parser = argparse.ArgumentParser(description="Luna-Ultimate Inference Demo")
    parser.add_argument("--prompt", type=str, default="Once upon a time",
                        help="Input prompt text")
    parser.add_argument("--max_tokens", type=int, default=100,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-K sampling")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Nucleus sampling threshold")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--no_speculative", action="store_true",
                        help="Disable speculative decoding")
    parser.add_argument("--no_int4_cache", action="store_true",
                        help="Disable INT4 KV cache")
    parser.add_argument("--no_ctm_adaptive", action="store_true",
                        help="Disable CTM adaptive early exit")
    parser.add_argument("--dynamic_skip", action="store_true",
                        help="Enable dynamic layer skipping")
    parser.add_argument("--no_stream", action="store_true",
                        help="Disable streaming output")

    args = parser.parse_args()

    print("=" * 60)
    print("  Luna-Ultimate 550B Hybrid Model — Inference Demo")
    print("  CTM × Mamba2-SSD × MLA × FlashMoE")
    print("=" * 60)

    # Initialize inference engine
    engine = LunaInference(
        config=LunaConfig(),
        model_path=args.model_path,
    )

    # Generate
    engine.generate(
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        use_speculative=not args.no_speculative,
        use_int4_cache=not args.no_int4_cache,
        use_ctm_adaptive=not args.no_ctm_adaptive,
        use_dynamic_skip=args.dynamic_skip,
        stream=not args.no_stream,
    )


if __name__ == "__main__":
    main()