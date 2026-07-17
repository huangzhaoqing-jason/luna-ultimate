"""SiFu-inspired white-box speech — imitate SJTU BriLLM (Zhao et al., arXiv:2503.11299).

Design (not a weight copy of BriLLM; structure imitation):
  - Each vocab token = one interpretable graph node (static semantic map)
  - Prediction = max signal energy after edge/node propagation (not black-box logits alone)
  - Every step emits a WhiteBoxTrace (node ids, energies, edges) for audit / creator control

Efficiency: shared edge linear + per-node modulators (low-rank stand-in for full VxV matrices).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TraceStep:
    position: int
    context_nodes: List[int]
    candidate_top: List[Tuple[int, float]]  # (node_id, energy)
    chosen_node: int
    chosen_energy: float
    attention: List[float]
    note: str = ""


@dataclass
class WhiteBoxTrace:
    """Fully inspectable reasoning path — the opposite of a black box."""

    steps: List[TraceStep] = field(default_factory=list)
    paradigm: str = "SiFu/BriLLM-inspired"
    paper: str = "arXiv:2503.11299"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "paradigm": self.paradigm,
            "paper": self.paper,
            "steps": [
                {
                    "position": s.position,
                    "context_nodes": s.context_nodes,
                    "candidate_top": s.candidate_top,
                    "chosen_node": s.chosen_node,
                    "chosen_energy": s.chosen_energy,
                    "attention": s.attention,
                    "note": s.note,
                }
                for s in self.steps
            ],
        }

    def explain(self, max_steps: int = 8) -> str:
        lines = [f"WhiteBox SiFu trace ({self.paper})"]
        for s in self.steps[:max_steps]:
            tops = ", ".join(f"{n}:{e:.3f}" for n, e in s.candidate_top[:5])
            lines.append(
                f"  t={s.position} ctx={s.context_nodes} -> node {s.chosen_node} "
                f"(E={s.chosen_energy:.3f}) top=[{tops}] {s.note}"
            )
        return "\n".join(lines)


class SiFuSpeech(nn.Module):
    """Token-as-node signal graph with energy-based next-node prediction."""

    def __init__(
        self,
        vocab_size: int = 512,
        d_node: int = 32,
        max_pos: int = 128,
        topk_trace: int = 8,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_node = d_node
        self.max_pos = max_pos
        self.topk_trace = topk_trace

        # Static semantic nodes: bias b_v (BriLLM node design)
        self.node_bias = nn.Embedding(vocab_size, d_node)
        # Edge modulators (efficient stand-in for W_{u,v})
        self.edge_src = nn.Embedding(vocab_size, d_node)
        self.edge_dst = nn.Embedding(vocab_size, d_node)
        self.edge_linear = nn.Linear(d_node, d_node, bias=True)
        # Context attention alphas over previous positions (learned)
        self.alpha_proj = nn.Linear(d_node, 1)
        # Bridge from brain state → initial signal seed
        self.state_to_signal = nn.Linear(d_node, d_node)

        pe = self._build_sinusoidal_pe(max_pos, d_node)
        self.register_buffer("pe", pe, persistent=False)

    @staticmethod
    def _build_sinusoidal_pe(max_pos: int, d: int) -> torch.Tensor:
        pos = torch.arange(max_pos).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe = torch.zeros(max_pos, d)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    def _edge_transform(self, r: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        """r: [B,d]; src/dst: [B] long — BriLLM-like W_src,dst · r + biases."""
        gate = torch.sigmoid(self.edge_src(src) * self.edge_dst(dst))
        return self.edge_linear(r * gate) + self.node_bias(dst)

    def encode_context(
        self, token_ids: torch.Tensor, seed: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        token_ids: [B, L]
        returns signals [B, L, d], last signal [B, d]
        """
        B, L = token_ids.shape
        device = token_ids.device
        if seed is None:
            r = torch.ones(B, self.d_node, device=device)
        else:
            r = torch.ones(B, self.d_node, device=device) + self.state_to_signal(seed)

        signals = []
        for i in range(L):
            tid = token_ids[:, i]
            if i == 0:
                r = F.gelu(r + self.node_bias(tid) + self.pe[i])
            else:
                prev = token_ids[:, i - 1]
                r = F.gelu(self._edge_transform(r, prev, tid) + self.pe[min(i, self.max_pos - 1)])
            signals.append(r)
        stacked = torch.stack(signals, dim=1)
        return stacked, stacked[:, -1]

    def energy_to_all(
        self,
        signals: torch.Tensor,
        token_ids: torch.Tensor,
        boost: Optional[torch.Tensor] = None,
        block_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute energy for every vocab node as next token.
        signals: [B, L, d], token_ids: [B, L]
        returns energies [B, V], attention [B, L]
        """
        B, L, d = signals.shape
        # attention over context positions
        alpha = self.alpha_proj(signals).squeeze(-1)  # [B, L]
        attn = torch.softmax(alpha, dim=-1)

        # Candidate nodes 0..V-1 — batch energy via low-rank expansion
        # For each context pos k and candidate u: e = || edge(signal_k, src=tok_k, dst=u) ||
        # Vectorized over u using dst embeddings
        V = self.vocab_size
        dst_all = torch.arange(V, device=signals.device)  # [V]
        edge_dst_all = self.edge_dst(dst_all)  # [V, d]
        node_bias_all = self.node_bias(dst_all)  # [V, d]

        energies = torch.zeros(B, V, device=signals.device)
        for k in range(L):
            r_k = signals[:, k, :]  # [B, d]
            src = token_ids[:, k]  # [B]
            src_e = self.edge_src(src)  # [B, d]
            # gate[b,v,d] = sigmoid(src_e[b] * edge_dst[v])
            gate = torch.sigmoid(src_e.unsqueeze(1) * edge_dst_all.unsqueeze(0))  # [B,V,d]
            transformed = self.edge_linear(r_k).unsqueeze(1) * gate + node_bias_all.unsqueeze(0)
            e = transformed.norm(dim=-1)  # [B, V]
            energies = energies + attn[:, k].unsqueeze(-1) * e

        if boost is not None:
            energies = energies + boost
        if block_mask is not None:
            energies = energies.masked_fill(block_mask, -1e9)
        return energies, attn

    def forward(
        self,
        token_ids: torch.Tensor,
        seed: Optional[torch.Tensor] = None,
        boost: Optional[torch.Tensor] = None,
        block_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        signals, last = self.encode_context(token_ids, seed=seed)
        energies, attn = self.energy_to_all(signals, token_ids, boost=boost, block_mask=block_mask)
        chosen = energies.argmax(dim=-1)
        return {
            "energies": energies,
            "attention": attn,
            "chosen": chosen,
            "signals": signals,
            "last_signal": last,
        }

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new: int = 8,
        seed: Optional[torch.Tensor] = None,
        boost: Optional[torch.Tensor] = None,
        block_mask: Optional[torch.Tensor] = None,
        forced_nodes: Optional[Sequence[int]] = None,
    ) -> Tuple[torch.Tensor, WhiteBoxTrace]:
        """Autoregressive SiFu generation with full white-box trace."""
        ids = prompt_ids.clone()
        B = ids.shape[0]
        if B != 1:
            raise ValueError("white-box generate supports batch=1 for clear traces")
        trace = WhiteBoxTrace()
        forced = list(forced_nodes or [])

        for t in range(max_new):
            out = self.forward(ids, seed=seed, boost=boost, block_mask=block_mask)
            energies = out["energies"][0]
            attn = out["attention"][0].tolist()
            topk = torch.topk(energies, k=min(self.topk_trace, self.vocab_size))
            cand = list(zip(topk.indices.tolist(), topk.values.tolist()))

            note = ""
            if forced:
                chosen = int(forced.pop(0))
                note = "creator_forced"
            else:
                chosen = int(energies.argmax().item())

            step = TraceStep(
                position=ids.shape[1],
                context_nodes=ids[0].tolist(),
                candidate_top=cand,
                chosen_node=chosen,
                chosen_energy=float(energies[chosen].item()),
                attention=attn,
                note=note,
            )
            trace.steps.append(step)
            ids = torch.cat(
                [ids, torch.tensor([[chosen]], device=ids.device, dtype=ids.dtype)], dim=1
            )
        return ids, trace
