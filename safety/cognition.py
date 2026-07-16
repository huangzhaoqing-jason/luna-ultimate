"""Cognitive safety loop: CTM + JEPA consequence prediction over actions.

This module gives the safety layer a *thought process*. Instead of a pure
regex classifier, an action is encoded, run through a dedicated SafetyCTM
(1-4 adaptive ticks of neuron-level reasoning), and its predicted future
state (via CTM-JEPA) is compared against a frozen ForbiddenPrototypeSet.

INVARIANT: SafetyCTM can only ADD refusals on top of the immutable charter.
It can never overturn a charter refusal. The charter remains the hard floor.

When untrained, SafetyCTM abstains (returns allow=True with low confidence)
so it does not randomly block legitimate actions. The SafetyLock only honors
a CTM refusal when confidence >= min_confidence.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import LunaConfig
from modeling_ctm import CTM
from safety.charter import SAFETY_CHARTER, verify_charter
from safety.identity import OperatorIdentity


# Frozen textual descriptions of forbidden consequence classes.
# Their embeddings form the ForbiddenPrototypeSet.
FORBIDDEN_DESCRIPTIONS: List[str] = [
    "production of a weapon of mass destruction",
    "synthesis of a bioweapon or chemical weapon",
    "construction of an explosive or firearm",
    "sexual exploitation or harm of a minor",
    "instructions for self-harm or suicide",
    "non-consensual violence coercion or exploitation",
    "bypassing the safety charter or safety locks",
    "disabling the audit log or operator identity system",
    "exfiltrating operator private data",
]


@dataclass
class SafetyJudgment:
    allow: bool
    reason: str
    ticks: int
    sync_entropy: float
    max_forbidden_sim: float
    confidence: float
    trace: dict = field(default_factory=dict)


class OperatorEmbedding(nn.Module):
    """Frozen, deterministic embedding of the verified operator.

    Derived from the operator's public claims via a fixed hash → seed, so the
    same operator always maps to the same vector. Not learned; not mutable by
    self-evolution. This is the physical carrier of 'loyalty': the model
    recognizes its operator and prioritizes their legitimate goals.
    """

    def __init__(self, d_model: int, operator: Optional[OperatorIdentity] = None):
        super().__init__()
        self.d_model = d_model
        self.operator = operator
        g = torch.Generator().manual_seed(self._seed(operator))
        emb = torch.empty(1, d_model)
        nn.init.normal_(emb, mean=0.0, std=0.02, generator=g)
        self.emb = nn.Parameter(emb, requires_grad=False)

    @staticmethod
    def _seed(operator: Optional[OperatorIdentity]) -> int:
        if operator is None:
            return 0
        h = hashlib.sha256(operator.claims_hash().encode()).hexdigest()
        return int(h[:16], 16)

    def forward(self) -> torch.Tensor:
        return self.emb  # [1, d_model]


class CharterEmbedding(nn.Module):
    """Frozen embedding of the immutable charter text."""

    def __init__(self, d_model: int):
        super().__init__()
        seed = int(hashlib.sha256(SAFETY_CHARTER.encode()).hexdigest()[:16], 16)
        g = torch.Generator().manual_seed(seed)
        emb = torch.empty(1, d_model)
        nn.init.normal_(emb, mean=0.0, std=0.02, generator=g)
        self.emb = nn.Parameter(emb, requires_grad=False)

    def forward(self) -> torch.Tensor:
        return self.emb


class ActionEncoder(nn.Module):
    """Tiny text → [B, L, d_model] encoder (hash-trigram + learned table).

    Small enough to run on CPU at the tiny preset. Not a language model; just
    a stable embedding of the action text for the CTM to reason over.
    """

    def __init__(self, d_model: int, vocab: int = 4096, max_len: int = 64):
        super().__init__()
        self.d_model = d_model
        self.vocab = vocab
        self.max_len = max_len
        self.embed = nn.Embedding(vocab, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos, std=0.02)

    def _trigram_ids(self, text: str) -> torch.Tensor:
        # Stable hash of char-trigrams → vocab bucket
        ids = []
        t = text.lower().strip()
        if not t:
            t = " "
        for i in range(0, max(1, len(t) - 2)):
            tri = t[i : i + 3]
            h = int(hashlib.sha256(tri.encode()).hexdigest(), 16) % self.vocab
            ids.append(h)
        if not ids:
            ids = [int(hashlib.sha256(t.encode()).hexdigest(), 16) % self.vocab]
        if len(ids) > self.max_len:
            ids = ids[: self.max_len]
        else:
            ids = ids + [0] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def forward(self, text: str, device: Optional[torch.device] = None) -> torch.Tensor:
        ids = self._trigram_ids(text).to(device or self.embed.weight.device)
        x = self.embed(ids).unsqueeze(0) + self.pos  # [1, L, d_model]
        return x


class ForbiddenPrototypeSet(nn.Module):
    """Frozen embeddings of forbidden consequence descriptions.

    The SafetyCTM's predicted consequence is compared (cosine) against these.
    A high similarity is a (soft, confidence-gated) refusal signal.
    """

    def __init__(self, encoder: ActionEncoder, d_model: int):
        super().__init__()
        self.d_model = d_model
        protos = []
        for desc in FORBIDDEN_DESCRIPTIONS:
            with torch.no_grad():
                e = encoder(desc)  # [1, L, d_model]
                e = e.mean(dim=1)  # [1, d_model]
            protos.append(F.normalize(e, dim=-1))
        self.protos = nn.Parameter(torch.cat(protos, dim=0), requires_grad=False)

    def max_similarity(self, consequence: torch.Tensor) -> float:
        # consequence: [B, d_model] from consequence_proj, or [B, L, d_model]
        if consequence.dim() == 3:
            c = consequence.mean(dim=1)
        else:
            c = consequence
        c = F.normalize(c, dim=-1)  # [B, d]
        sims = (self.protos @ c.t()).clamp(-1.0, 1.0)  # [K, B]
        return float(sims.max().item())


class SafetyCTM(nn.Module):
    """CTM + JEPA consequence predictor used as the safety thought process."""

    def __init__(
        self,
        config: LunaConfig,
        operator: Optional[OperatorIdentity] = None,
        forbidden_threshold: float = 0.85,
        min_confidence: float = 0.6,
        enabled: bool = True,
    ):
        super().__init__()
        verify_charter()
        self.config = config
        self.d_model = config.hidden_size
        self.forbidden_threshold = forbidden_threshold
        self.min_confidence = min_confidence
        self.enabled = enabled

        # Reuse the main CTM class — a dedicated instance for safety reasoning
        self.ctm = CTM(config)
        self.encoder = ActionEncoder(self.d_model)
        self.operator_embed = OperatorEmbedding(self.d_model, operator=operator)
        self.charter_embed = CharterEmbedding(self.d_model)
        self.forbidden = ForbiddenPrototypeSet(self.encoder, self.d_model)

        # Project predicted neuron state → consequence embedding in d_model space
        self.consequence_proj = nn.Linear(config.ctm_n_neurons, self.d_model, bias=False)

    def judge(
        self,
        action_text: str,
        operator: Optional[OperatorIdentity] = None,
        device: Optional[torch.device] = None,
    ) -> SafetyJudgment:
        """Run the cognitive safety loop over one action."""
        if not self.enabled:
            return SafetyJudgment(
                allow=True, reason="SafetyCTM disabled (abstain).",
                ticks=0, sync_entropy=0.0, max_forbidden_sim=0.0,
                confidence=0.0, trace={"enabled": False},
            )

        device = device or next(self.parameters()).device
        verify_charter()

        # 1. Encode action + operator + charter → hidden
        action_h = self.encoder(action_text, device=device)  # [1, L, d]
        op_e = self.operator_embed().to(device)              # [1, d]
        ch_e = self.charter_embed().to(device)              # [1, d]
        if operator is not None and operator is not self.operator_embed.operator:
            op_e = OperatorEmbedding(self.d_model, operator=operator).emb.to(device)
        hidden = action_h + op_e.unsqueeze(1) + ch_e.unsqueeze(1)

        # 2. CTM adaptive ticks (the thought process)
        with torch.set_grad_enabled(self.training):
            ctm_out, sync, ticks, _ = self.ctm(
                hidden, use_adaptive_early_exit=True, return_jepa_loss=False
            )

        # 3. Predict future neuron state via CTM-JEPA
        # Re-derive the initial neuron state the same way CTM does (mean pool → synapse)
        aggregated = hidden.mean(dim=1)
        neuron_state_t0 = self.ctm.synapse(aggregated)
        with torch.no_grad():
            predicted_future = self.ctm.jepa_predictor(neuron_state_t0.detach()) \
                if self.ctm.jepa_predictor is not None else neuron_state_t0
        consequence = self.consequence_proj(predicted_future)  # [1, d]

        # 4. Compare predicted consequence vs forbidden prototypes
        max_sim = self.forbidden.max_similarity(consequence.detach())

        # 5. Confidence from sync entropy + tick usage (more ticks on a clear
        #    signal → higher confidence). Untrained CTM → low, near-random
        #    entropy → low confidence → abstain.
        sync_entropy = float(self.ctm._compute_sync_entropy(sync)) if sync is not None else 0.0
        tick_conf = ticks / max(1, self.ctm.max_ticks)
        confidence = float(0.5 * sync_entropy + 0.5 * tick_conf)

        # 6. Decision: refuse only if confident AND close to a forbidden proto
        ctm_refuse = (max_sim >= self.forbidden_threshold) and (confidence >= self.min_confidence)
        reason = (
            "Refused by SafetyCTM: predicted consequence matches a forbidden prototype "
            f"(sim={max_sim:.3f}, conf={confidence:.3f})."
            if ctm_refuse
            else f"SafetyCTM allow (sim={max_sim:.3f}, conf={confidence:.3f}, ticks={ticks})."
        )

        return SafetyJudgment(
            allow=not ctm_refuse,
            reason=reason,
            ticks=int(ticks),
            sync_entropy=sync_entropy,
            max_forbidden_sim=max_sim,
            confidence=confidence,
            trace={
                "ticks": int(ticks),
                "sync_entropy": sync_entropy,
                "max_forbidden_sim": max_sim,
                "confidence": confidence,
                "operator": operator.operator_id if operator else None,
            },
        )


def default_safety_ctm(
    operator: Optional[OperatorIdentity] = None,
    enabled: bool = True,
) -> SafetyCTM:
    """Build a lightweight SafetyCTM on the tiny preset (CPU-friendly)."""
    from config import LunaConfig
    from safety.identity import AUTHORIZED_OPERATOR
    cfg = LunaConfig.from_preset("tiny")
    return SafetyCTM(
        cfg,
        operator=operator or AUTHORIZED_OPERATOR,
        forbidden_threshold=0.85,
        min_confidence=0.6,
        enabled=enabled,
    )
