"""Short-train + proxy eval + cost model fitness for Luna Evolve."""

from __future__ import annotations

import time
from typing import Optional

import torch
import torch.nn.functional as F

from cost_model import estimate_cost
from evolve.archive import Individual
from evolve.genome import Genome
from modeling_luna_ultimate import LunaUltimateFused


def _proxy_quality(ce_loss: float, avg_ticks: float, max_ticks: int, vocab_size: int) -> float:
    """Map CE + tick efficiency to quality in ~[0, 1].

    Random-init CE ≈ ln(vocab); score improves as CE drops below that baseline.
    """
    import math
    baseline = math.log(max(2, vocab_size))
    # 0 at baseline, →1 as CE → 0
    ce_score = max(0.0, min(1.0, (baseline - ce_loss) / max(baseline, 1e-6) + 0.5))
    # Soften: use logistic around baseline
    ce_score = 1.0 / (1.0 + math.exp((ce_loss - baseline)))
    tick_eff = 1.0 - (avg_ticks / max(1, max_ticks)) * 0.2
    return max(0.0, min(1.0, 0.85 * ce_score + 0.15 * max(0.0, tick_eff)))


def evaluate_genome(
    genome: Genome,
    train_steps: int = 5,
    batch_size: int = 2,
    seq_len: int = 64,
    device: Optional[torch.device] = None,
    allow_large: bool = False,
) -> Individual:
    """Short train on random tokens + cost estimate → Individual."""
    device = device or torch.device("cpu")
    config = genome.to_config(allow_large=allow_large)
    config.max_steps = train_steps

    model = LunaUltimateFused(config).to(device)
    model.set_training_stage(genome.train.stage)
    model.train()

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=0.0,
    )

    t0 = time.time()
    last_ce = 0.0
    last_ticks = 0.0

    for step in range(train_steps):
        ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
        labels = ids.clone()
        labels[:, :-1] = ids[:, 1:]
        labels[:, -1] = -100

        outputs = model(
            ids,
            use_ctm_adaptive=genome.infer.early_exit,
            use_dynamic_skip=genome.infer.layer_skip,
            return_all_losses=True,
        )
        loss, stats = model.compute_loss(
            outputs, labels, step=step, total_steps=train_steps
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        last_ce = stats.get("lm_loss_raw", float(loss.detach().item()))
        last_ticks = stats.get("avg_ticks", float(model.last_avg_ticks))

    latency_ms = (time.time() - t0) * 1000.0 / max(1, train_steps)

    # Eval forward (no grad) for latency proxy
    model.eval()
    with torch.no_grad():
        ids = torch.randint(0, config.vocab_size, (1, seq_len), device=device)
        t1 = time.time()
        _ = model(ids, use_ctm_adaptive=genome.infer.early_exit, return_all_losses=False)
        latency_ms = 0.5 * latency_ms + 0.5 * (time.time() - t1) * 1000.0

    cost = estimate_cost(config, batch_size=batch_size, seq_len=seq_len)
    quality = _proxy_quality(last_ce, last_ticks, config.ctm_max_ticks, config.vocab_size)

    # Agent genome: reweight quality slightly by self-declared eval mix (smoke)
    w = (
        genome.agent.eval_weight_gsm8k
        + genome.agent.eval_weight_code
        + genome.agent.eval_weight_mmlu
        + genome.agent.eval_weight_bbh
    )
    quality = quality * (0.95 + 0.05 * min(1.0, w))

    # 三层安全门：red-team AND cognitive AND values（不合格基因组不可晋升）
    from evolve.safety_fitness import gate_fitness, safety_score, cognitive_score, values_score
    s = safety_score()
    c = cognitive_score()
    v = values_score()
    gated = gate_fitness(quality, safety=s, cognitive=c, values=v)

    return Individual(
        genome=genome,
        quality=gated,
        active_flops=cost.active_flops_per_token,
        peak_vram_gb=cost.peak_vram_gb,
        latency_ms=latency_ms,
        ce_loss=last_ce,
        avg_ticks=last_ticks,
        dollars_per_mtok=cost.dollars_per_million_tokens,
        meta={
            "total_params_b": cost.total_params_b,
            "active_params_b": cost.active_params_b,
            "preset": genome.preset,
            "raw_quality": quality,
            "safety_score": s,
            "cognitive_score": c,
            "values_score": v,
        },
    )
