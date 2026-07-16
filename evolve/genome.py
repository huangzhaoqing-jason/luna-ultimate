"""Serializable genomes for Luna Evolve."""

from __future__ import annotations

import copy
import json
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from config import LunaConfig


@dataclass
class ArchGenome:
    mamba_ratio: float = 0.5  # fraction of layers that are Mamba
    num_routed_experts: int = 4
    top_k: int = 2
    ctm_max_ticks: int = 2
    ctm_inject_every: int = 2
    kv_lora_rank: int = 64
    intermediate_size: int = 512
    hidden_size: Optional[int] = None  # None = keep preset


@dataclass
class TrainGenome:
    lr_multiplier: float = 1.0
    archer_knowledge_kl: float = 0.1
    archer_reasoning_kl: float = 0.001
    moe_aux_coeff: float = 0.01
    moe_z_coeff: float = 0.001
    stage: int = 1


@dataclass
class InferGenome:
    early_exit: bool = True
    layer_skip: bool = False
    layer_skip_threshold: float = 0.7
    draft_tokens: int = 0
    kv_int4: bool = True


@dataclass
class AgentGenome:
    eval_weight_gsm8k: float = 0.25
    eval_weight_code: float = 0.25
    eval_weight_mmlu: float = 0.25
    eval_weight_bbh: float = 0.25
    self_play_ratio: float = 0.0
    distill_teacher: str = "none"  # none | self | external


@dataclass
class Genome:
    preset: str = "tiny"
    arch: ArchGenome = field(default_factory=ArchGenome)
    train: TrainGenome = field(default_factory=TrainGenome)
    infer: InferGenome = field(default_factory=InferGenome)
    agent: AgentGenome = field(default_factory=AgentGenome)
    generation: int = 0
    parent_id: Optional[str] = None
    genome_id: str = "seed"

    def clone(self) -> "Genome":
        g = copy.deepcopy(self)
        g.parent_id = self.genome_id
        return g

    def to_dict(self) -> Dict[str, Any]:
        return {
            "preset": self.preset,
            "arch": asdict(self.arch),
            "train": asdict(self.train),
            "infer": asdict(self.infer),
            "agent": asdict(self.agent),
            "generation": self.generation,
            "parent_id": self.parent_id,
            "genome_id": self.genome_id,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Genome":
        return cls(
            preset=data.get("preset", "tiny"),
            arch=ArchGenome(**data.get("arch", {})),
            train=TrainGenome(**data.get("train", {})),
            infer=InferGenome(**data.get("infer", {})),
            agent=AgentGenome(**data.get("agent", {})),
            generation=data.get("generation", 0),
            parent_id=data.get("parent_id"),
            genome_id=data.get("genome_id", "seed"),
        )

    @classmethod
    def from_preset(cls, preset: str) -> "Genome":
        cfg = LunaConfig.from_preset(preset)
        ratio = cfg.mamba2_layers / max(1, cfg.num_hidden_layers)
        return cls(
            preset=preset,
            arch=ArchGenome(
                mamba_ratio=ratio,
                num_routed_experts=cfg.num_routed_experts,
                top_k=cfg.num_expert_activated,
                ctm_max_ticks=cfg.ctm_max_ticks,
                ctm_inject_every=cfg.ctm_inject_every,
                kv_lora_rank=cfg.kv_lora_rank,
                intermediate_size=cfg.intermediate_size,
                hidden_size=None,
            ),
            train=TrainGenome(
                archer_knowledge_kl=cfg.archer_knowledge_kl_weight,
                archer_reasoning_kl=cfg.archer_reasoning_kl_weight,
                moe_aux_coeff=cfg.moe_aux_loss_coeff,
                moe_z_coeff=cfg.moe_z_loss_coeff,
            ),
            infer=InferGenome(
                early_exit=cfg.use_ctm_adaptive_early_exit,
                layer_skip=cfg.use_dynamic_layer_skip,
                layer_skip_threshold=cfg.layer_skip_prob_threshold,
                draft_tokens=cfg.num_draft_tokens if cfg.use_speculative_decoding else 0,
                kv_int4=cfg.use_kv_cache_int4,
            ),
            genome_id=f"{preset}-seed",
        )

    def to_config(self, allow_large: bool = False) -> LunaConfig:
        """Materialize a LunaConfig from preset + arch/train/infer genes."""
        if self.preset in ("550b", "77b_active") and not allow_large:
            raise RuntimeError(
                "Refusing to materialize 550b/77b_active inside evolve without "
                "--allow-large. Stabilize genomes on tiny/1b/7b first."
            )

        cfg = LunaConfig.from_preset(self.preset)
        n_layers = cfg.num_hidden_layers
        mamba = max(1, min(n_layers - 1, int(round(self.arch.mamba_ratio * n_layers))))
        mla = n_layers - mamba

        # Keep ranks divisible / within bounds
        kv_rank = max(16, min(self.arch.kv_lora_rank, cfg.hidden_size))
        if kv_rank % 2 == 1:
            kv_rank += 1
        top_k = max(1, min(self.arch.top_k, self.arch.num_routed_experts))
        experts = max(top_k, self.arch.num_routed_experts)

        overrides = dict(
            mamba2_layers=mamba,
            mla_layers=mla,
            num_routed_experts=experts,
            num_expert_activated=top_k,
            ctm_max_ticks=max(1, self.arch.ctm_max_ticks),
            ctm_inject_every=max(1, self.arch.ctm_inject_every),
            kv_lora_rank=kv_rank,
            intermediate_size=max(64, self.arch.intermediate_size),
            learning_rate=cfg.learning_rate * self.train.lr_multiplier,
            archer_knowledge_kl_weight=self.train.archer_knowledge_kl,
            archer_reasoning_kl_weight=self.train.archer_reasoning_kl,
            moe_aux_loss_coeff=self.train.moe_aux_coeff,
            moe_z_loss_coeff=self.train.moe_z_coeff,
            use_ctm_adaptive_early_exit=self.infer.early_exit,
            use_dynamic_layer_skip=self.infer.layer_skip,
            layer_skip_prob_threshold=self.infer.layer_skip_threshold,
            use_kv_cache_int4=self.infer.kv_int4,
            use_speculative_decoding=self.infer.draft_tokens > 0,
            num_draft_tokens=max(0, self.infer.draft_tokens),
            vjepa_enabled=False if self.train.stage < 3 else cfg.vjepa_enabled,
        )
        return LunaConfig.from_preset(self.preset, **overrides)


def mutate(genome: Genome, rng: Optional[random.Random] = None) -> Genome:
    rng = rng or random.Random()
    child = genome.clone()
    child.generation = genome.generation + 1
    child.genome_id = f"g{child.generation}-{rng.randrange(1_000_000):06d}"

    # Arch mutations
    if rng.random() < 0.5:
        child.arch.mamba_ratio = min(0.9, max(0.1, child.arch.mamba_ratio + rng.uniform(-0.15, 0.15)))
    if rng.random() < 0.4:
        child.arch.num_routed_experts = max(2, child.arch.num_routed_experts + rng.choice([-2, -1, 1, 2]))
    if rng.random() < 0.4:
        child.arch.top_k = max(1, min(child.arch.num_routed_experts, child.arch.top_k + rng.choice([-1, 1])))
    if rng.random() < 0.4:
        child.arch.ctm_max_ticks = max(1, min(4, child.arch.ctm_max_ticks + rng.choice([-1, 1])))
    if rng.random() < 0.3:
        child.arch.ctm_inject_every = max(1, min(8, child.arch.ctm_inject_every + rng.choice([-1, 1])))
    if rng.random() < 0.3:
        child.arch.kv_lora_rank = max(16, int(child.arch.kv_lora_rank * rng.uniform(0.75, 1.25)))
        if child.arch.kv_lora_rank % 2:
            child.arch.kv_lora_rank += 1
    if rng.random() < 0.3:
        child.arch.intermediate_size = max(64, int(child.arch.intermediate_size * rng.uniform(0.8, 1.2)))

    # Train mutations
    if rng.random() < 0.4:
        child.train.lr_multiplier = min(3.0, max(0.2, child.train.lr_multiplier * rng.uniform(0.7, 1.4)))
    if rng.random() < 0.3:
        child.train.moe_aux_coeff = min(0.1, max(1e-4, child.train.moe_aux_coeff * rng.uniform(0.5, 2.0)))
    if rng.random() < 0.3:
        child.train.stage = rng.choice([1, 2])

    # Infer mutations
    if rng.random() < 0.3:
        child.infer.early_exit = not child.infer.early_exit
    if rng.random() < 0.3:
        child.infer.kv_int4 = not child.infer.kv_int4
    if rng.random() < 0.3:
        child.infer.layer_skip_threshold = min(
            0.95, max(0.4, child.infer.layer_skip_threshold + rng.uniform(-0.1, 0.1))
        )

    # Agent mutations
    if rng.random() < 0.3:
        weights = [
            child.agent.eval_weight_gsm8k,
            child.agent.eval_weight_code,
            child.agent.eval_weight_mmlu,
            child.agent.eval_weight_bbh,
        ]
        i = rng.randrange(4)
        weights[i] = max(0.05, weights[i] + rng.uniform(-0.1, 0.1))
        s = sum(weights)
        (
            child.agent.eval_weight_gsm8k,
            child.agent.eval_weight_code,
            child.agent.eval_weight_mmlu,
            child.agent.eval_weight_bbh,
        ) = [w / s for w in weights]

    return child


def crossover(a: Genome, b: Genome, rng: Optional[random.Random] = None) -> Genome:
    rng = rng or random.Random()
    child = a.clone()
    child.generation = max(a.generation, b.generation) + 1
    child.genome_id = f"x{child.generation}-{rng.randrange(1_000_000):06d}"
    child.parent_id = f"{a.genome_id}+{b.genome_id}"
    if rng.random() < 0.5:
        child.arch = copy.deepcopy(b.arch)
    if rng.random() < 0.5:
        child.train = copy.deepcopy(b.train)
    if rng.random() < 0.5:
        child.infer = copy.deepcopy(b.infer)
    if rng.random() < 0.5:
        child.agent = copy.deepcopy(b.agent)
    return child


def seed_population(preset: str, n: int, rng: Optional[random.Random] = None) -> List[Genome]:
    rng = rng or random.Random()
    pop = [Genome.from_preset(preset)]
    while len(pop) < n:
        pop.append(mutate(pop[0], rng))
    return pop
