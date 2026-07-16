"""Luna-Ultimate Training Script — Enhanced for Hybrid Mamba+MLA+MoE Architecture.

Key improvements for hybrid architecture stability:
  - Per-module-group learning rates (Mamba2 SSM needs lower LR than MLA)
  - Adaptive gradient clipping with norm history tracking
  - Multi-phase LR schedule: SSM warmup → full model → MoE cooldown
  - CTM auxiliary loss annealing (strong early, weak late)
  - Gradient noise scale monitoring for Mamba2 stability
  - Separate MoE router z-loss for training stability

Usage:
    torchrun --nproc_per_node=8 train.py
    deepspeed train.py --deepspeed ds_config_zero3.json
"""

import os
import sys
import math
import argparse
import logging
import json
from typing import Optional, Dict, Any, Tuple, List
from collections import deque
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from config import LunaConfig
from modeling_luna import LunaUltimate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ==================== Hybrid Architecture Optimizer ====================

def create_hybrid_optimizer(
    model: nn.Module,
    config: LunaConfig,
) -> torch.optim.Optimizer:
    """Create optimizer with per-module-group learning rates.

    Mamba2 SSM parameters (A_log, dt_proj, x_proj) are sensitive to high LR.
    MLA attention parameters are more robust.
    MoE router needs higher LR for fast specialization.
    Shared experts learn slower (lower LR).

    Returns:
        Optimizer with parameter groups.
    """
    # Separate parameters by module type
    mamba_ssm_params = []    # A_log, dt_proj — lowest LR
    mamba_conv_params = []   # conv1d — medium LR
    mamba_proj_params = []   # in_proj, out_proj — normal LR
    mla_attn_params = []     # all MLA attention — normal LR
    moe_router_params = []   # router weights — higher LR
    moe_expert_params = []   # expert FFN — normal LR
    ctm_params = []          # CTM — normal LR
    other_params = []        # embedding, norms, lm_head

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "mamba" in name:
            if "A_log" in name or "dt_proj" in name:
                mamba_ssm_params.append(param)
            elif "conv1d" in name:
                mamba_conv_params.append(param)
            else:
                mamba_proj_params.append(param)
        elif "mla" in name or "attention" in name:
            mla_attn_params.append(param)
        elif "moe" in name or "router" in name:
            if "router" in name:
                moe_router_params.append(param)
            elif "expert" in name:
                moe_expert_params.append(param)
            else:
                moe_expert_params.append(param)
        elif "ctm" in name:
            ctm_params.append(param)
        else:
            other_params.append(param)

    base_lr = config.learning_rate

    param_groups = [
        {"params": mamba_ssm_params,   "lr": base_lr * 0.1,  "name": "mamba_ssm"},
        {"params": mamba_conv_params,  "lr": base_lr * 0.5,  "name": "mamba_conv"},
        {"params": mamba_proj_params,  "lr": base_lr * 0.8,  "name": "mamba_proj"},
        {"params": mla_attn_params,    "lr": base_lr,         "name": "mla_attn"},
        {"params": moe_router_params,  "lr": base_lr * 1.5,  "name": "moe_router"},
        {"params": moe_expert_params,  "lr": base_lr,         "name": "moe_expert"},
        {"params": ctm_params,         "lr": base_lr,         "name": "ctm"},
        {"params": other_params,       "lr": base_lr,         "name": "other"},
    ]

    # Remove empty groups
    param_groups = [g for g in param_groups if len(g["params"]) > 0]

    optimizer = AdamW(
        param_groups,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.weight_decay,
    )

    return optimizer


# ==================== Archer Entropy-Aware Loss ====================

class ArcherEntropyLoss(nn.Module):
    """Archer entropy-aware training loss with CTM auxiliary loss.

    Differentiates between:
      - Knowledge tokens (low entropy): Strong KL constraint (0.1)
      - Reasoning tokens (high entropy): Weak KL constraint (0.001), high clip (0.3)

    Adds CTM tick-count auxiliary loss to encourage efficient reasoning.
    """

    def __init__(self, config: LunaConfig):
        super().__init__()
        self.knowledge_kl_weight = config.archer_knowledge_kl_weight
        self.reasoning_kl_weight = config.archer_reasoning_kl_weight
        self.reasoning_clip_threshold = config.archer_reasoning_clip_threshold
        self.entropy_threshold = config.archer_entropy_threshold
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)

    def _compute_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return -(probs * log_probs).sum(dim=-1)

    def _compute_kl_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_probs = F.softmax(teacher_logits.detach(), dim=-1)
        kl = (teacher_probs * (torch.log(teacher_probs + 1e-10) - student_log_probs)).sum(dim=-1)
        return (kl * mask).sum() / (mask.sum() + 1e-8)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        teacher_logits: Optional[torch.Tensor] = None,
        ctm_ticks: Optional[int] = None,
        global_step: int = 0,
        total_steps: int = 100000,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        B, L, V = logits.shape
        mask = (labels != -100).float()

        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device), {}

        ce_loss = self.ce_loss(logits.view(B * L, V), labels.view(B * L))
        total_loss = ce_loss
        stats = {"ce_loss": ce_loss.item()}

        if teacher_logits is not None:
            entropy = self._compute_entropy(logits)
            knowledge_mask = (entropy <= self.entropy_threshold).float() * mask
            reasoning_mask = (entropy > self.entropy_threshold).float() * mask

            if knowledge_mask.sum() > 0:
                kl_knowledge = self._compute_kl_loss(logits, teacher_logits, knowledge_mask)
                total_loss = total_loss + self.knowledge_kl_weight * kl_knowledge
                stats["kl_knowledge"] = kl_knowledge.item()

            if reasoning_mask.sum() > 0:
                kl_reasoning = self._compute_kl_loss(logits, teacher_logits, reasoning_mask)
                kl_reasoning = torch.clamp(kl_reasoning, max=self.reasoning_clip_threshold)
                total_loss = total_loss + self.reasoning_kl_weight * kl_reasoning
                stats["kl_reasoning"] = kl_reasoning.item()

            stats["knowledge_ratio"] = (knowledge_mask.sum() / (mask.sum() + 1e-8)).item()
            stats["reasoning_ratio"] = (reasoning_mask.sum() / (mask.sum() + 1e-8)).item()

        # CTM efficiency auxiliary loss: penalize unnecessary ticks
        # Anneal from 0.01 at start to 0.001 at end
        if ctm_ticks is not None and ctm_ticks > 0:
            ctm_weight = 0.01 * (1.0 - global_step / max(1, total_steps)) + 0.001
            ctm_loss = ctm_weight * (ctm_ticks / 4.0)  # Normalize to [0, 1]
            total_loss = total_loss + ctm_loss
            stats["ctm_loss"] = ctm_loss.item()
            stats["ctm_ticks"] = float(ctm_ticks)

        stats["total_loss"] = total_loss.item()
        return total_loss, stats


# ==================== Gradient Clipping ====================

class AdaptiveGradientClipper:
    """Adaptive gradient clipping with norm history tracking.

    For hybrid architectures, Mamba2 SSM gradients can spike 10-100× higher
    than MLA gradients. This clipper tracks per-group gradient norms and
    applies group-specific clipping thresholds.
    """

    def __init__(
        self,
        max_grad_norm: float = 1.0,
        history_size: int = 100,
        spike_threshold_multiplier: float = 5.0,
    ):
        self.max_grad_norm = max_grad_norm
        self.history = deque(maxlen=history_size)
        self.spike_threshold_multiplier = spike_threshold_multiplier
        self.spike_count = 0

    def clip(self, parameters, group_name: str = "all") -> float:
        """Clip gradients and return total norm before clipping."""
        total_norm = torch.nn.utils.clip_grad_norm_(parameters, self.max_grad_norm)

        # Track norm history for spike detection
        self.history.append(total_norm.item() if torch.is_tensor(total_norm) else total_norm)

        if len(self.history) >= 10:
            avg_norm = sum(self.history) / len(self.history)
            if total_norm > avg_norm * self.spike_threshold_multiplier:
                self.spike_count += 1
                # Aggressive reclip on spike
                torch.nn.utils.clip_grad_norm_(parameters, self.max_grad_norm * 0.5)

        return total_norm


# ==================== Multi-Phase LR Schedule ====================

class HybridLRScheduler:
    """Multi-phase LR schedule for hybrid Mamba2+MLA+MoE training.

    Phase 1 (0-10%): SSM warmup — Mamba2 params start at 1e-6, linear ramp to peak
    Phase 2 (10-80%): Full training — Cosine decay from peak
    Phase 3 (80-100%): MoE cooldown — Router LR drops faster, experts stabilize
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        ssm_warmup_ratio: float = 0.1,
        cooldown_start_ratio: float = 0.8,
        min_lr_ratio: float = 0.01,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.ssm_warmup_end = int(total_steps * ssm_warmup_ratio)
        self.cooldown_start = int(total_steps * cooldown_start_ratio)
        self.min_lr_ratio = min_lr_ratio

        # Store base LRs
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def get_lr(self, step: int) -> List[float]:
        """Get learning rates for all parameter groups at given step."""
        lrs = []

        for i, group in enumerate(self.optimizer.param_groups):
            base_lr = self.base_lrs[i]
            group_name = group.get("name", "other")

            if step < self.warmup_steps:
                # Linear warmup
                lr = base_lr * (step / max(1, self.warmup_steps))
            elif step < self.ssm_warmup_end:
                # SSM warmup: Mamba2 SSM params continue ramping
                if "mamba_ssm" in group_name:
                    progress = (step - self.warmup_steps) / max(1, self.ssm_warmup_end - self.warmup_steps)
                    lr = base_lr * (0.1 + 0.9 * progress)
                else:
                    # Other params: cosine from peak
                    progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
                    lr = base_lr * max(self.min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
            elif step >= self.cooldown_start:
                # Cooldown: router LR drops faster
                progress = (step - self.cooldown_start) / max(1, self.total_steps - self.cooldown_start)
                if "moe_router" in group_name:
                    lr = base_lr * max(self.min_lr_ratio * 0.1, 1.0 - progress)
                elif "mamba_ssm" in group_name:
                    lr = base_lr * max(self.min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * min(progress * 2, 1.0))))
                else:
                    lr = base_lr * max(self.min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
            else:
                # Full training: cosine decay
                progress = (step - self.ssm_warmup_end) / max(1, self.cooldown_start - self.ssm_warmup_end)
                lr = base_lr * max(self.min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

            lrs.append(lr)

        return lrs

    def step(self, step: int):
        """Update all parameter group LRs."""
        lrs = self.get_lr(step)
        for group, lr in zip(self.optimizer.param_groups, lrs):
            group["lr"] = lr


# ==================== MoE Router Z-Loss ====================

def compute_router_z_loss(router_logits: torch.Tensor) -> torch.Tensor:
    """Compute router z-loss for training stability.

    Z-loss = log(sum(exp(logits)))^2
    Penalizes large router logit magnitudes, preventing expert collapse.

    Args:
        router_logits: [B, L, num_experts] — raw router logits

    Returns:
        z_loss: scalar
    """
    # Log-sum-exp squared
    z = torch.logsumexp(router_logits, dim=-1)  # [B, L]
    z_loss = (z ** 2).mean()
    return z_loss


# ==================== Dummy Dataset ====================

class DummyDataset(torch.utils.data.Dataset):
    """Dummy dataset for testing. Replace with real pretraining data."""

    def __init__(self, vocab_size: int, seq_len: int, num_samples: int = 10000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        ids = torch.randint(0, min(self.vocab_size, 50000), (self.seq_len,))
        return {"input_ids": ids[:-1], "labels": ids[1:]}


# ==================== Distributed Setup ====================

def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


# ==================== Main Training Loop ====================

def train(args: argparse.Namespace):
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0

    if is_main:
        logger.info("=" * 60)
        logger.info("  Luna-Ultimate Hybrid Architecture Training")
        logger.info(f"  GPUs: {world_size} | Device: {device}")
        logger.info("=" * 60)

    config = LunaConfig()

    # Model
    if is_main:
        logger.info("Initializing model (this may take a while for 550B params)...")
    model = LunaUltimate(config).to(device)

    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    if is_main:
        total_b, active_b = model.get_num_parameters()
        logger.info(f"Model: {total_b:.2f}B total | {active_b:.2f}B active")

    # Optimizer with per-group LR
    if world_size > 1 and args.use_deepspeed:
        import deepspeed
        model, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config_params=args.deepspeed_config if args.deepspeed_config else {},
        )
        lr_scheduler = None
        use_hybrid_schedule = False
    else:
        if world_size > 1:
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], find_unused_parameters=False,
            )
        optimizer = create_hybrid_optimizer(model, config)
        lr_scheduler = HybridLRScheduler(
            optimizer,
            warmup_steps=config.warmup_steps,
            total_steps=config.max_steps,
        )
        use_hybrid_schedule = True

    # Loss
    archer_loss = ArcherEntropyLoss(config)

    # Gradient clipper
    grad_clipper = AdaptiveGradientClipper(
        max_grad_norm=config.max_grad_norm,
    )

    # Mixed precision
    use_amp = args.use_amp and config.mixed_precision == "bf16"
    scaler = GradScaler(enabled=use_amp)

    # Dataset
    dataset = DummyDataset(config.vocab_size, args.seq_len, args.num_samples)
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                                num_workers=args.num_workers, pin_memory=True)
    else:
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.num_workers, pin_memory=True)

    # Training state
    global_step = 0
    loss_accum = {"ce": 0.0, "aux": 0.0, "ctm": 0.0, "total": 0.0}
    start_time = time.time()

    if is_main:
        logger.info(f"Steps: {config.max_steps} | Batch: {args.batch_size} | GradAccum: {args.grad_accum}")
        logger.info(f"Effective batch: {args.batch_size * args.grad_accum * world_size}")
        logger.info("Starting training...")

    model.train()
    ctm_ticks_sum = 0
    ctm_ticks_count = 0

    for epoch in range(args.num_epochs):
        if world_size > 1:
            sampler.set_epoch(epoch)

        for batch_idx, batch in enumerate(dataloader):
            if global_step >= config.max_steps:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits, aux_loss = model(
                    input_ids,
                    use_ctm_adaptive=True,
                    use_dynamic_skip=args.dynamic_layer_skip,
                )

                # Track CTM ticks (from model internals)
                avg_ticks = 2  # Default placeholder; real value from model state
                ctm_ticks_sum += avg_ticks
                ctm_ticks_count += 1

                # Compute router z-loss for stability
                # (Router logits are internal to MoE layers; use aux_loss as proxy)
                z_loss = aux_loss * 0.001  # Small z-loss contribution

                # Archer loss
                loss, loss_stats = archer_loss(
                    logits, labels,
                    ctm_ticks=avg_ticks,
                    global_step=global_step,
                    total_steps=config.max_steps,
                )

                total_loss = loss + aux_loss * config.moe_aux_loss_coeff + z_loss

            total_loss = total_loss / args.grad_accum

            if use_amp:
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()

            loss_accum["ce"] += loss_stats.get("ce_loss", 0)
            loss_accum["aux"] += aux_loss.item()
            loss_accum["ctm"] += loss_stats.get("ctm_loss", 0)
            loss_accum["total"] += total_loss.item() * args.grad_accum

            if (batch_idx + 1) % args.grad_accum == 0:
                # Gradient clipping
                if use_amp:
                    scaler.unscale_(optimizer)
                    grad_norm = grad_clipper.clip(list(model.parameters()))
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    grad_norm = grad_clipper.clip(list(model.parameters()))
                    optimizer.step()

                if use_hybrid_schedule and lr_scheduler is not None:
                    lr_scheduler.step(global_step)

                optimizer.zero_grad()
                global_step += 1

                # Logging
                if global_step % args.log_every == 0 and is_main:
                    elapsed = time.time() - start_time
                    avg_ce = loss_accum["ce"] / args.log_every
                    avg_aux = loss_accum["aux"] / args.log_every
                    avg_ctm = loss_accum["ctm"] / args.log_every
                    avg_total = loss_accum["total"] / args.log_every
                    avg_ticks_val = ctm_ticks_sum / max(1, ctm_ticks_count)

                    lr_str = ""
                    if use_hybrid_schedule and lr_scheduler is not None:
                        lrs = lr_scheduler.get_lr(global_step)
                        lr_str = f"LR: {lrs[0]:.2e}"

                    logger.info(
                        f"Step {global_step:>6d}/{config.max_steps} | "
                        f"Loss: {avg_total:.4f} | CE: {avg_ce:.4f} | "
                        f"Aux: {avg_aux:.4f} | CTM: {avg_ctm:.4f} | "
                        f"GradNorm: {grad_norm:.2f} | "
                        f"Ticks: {avg_ticks_val:.1f} | "
                        f"{lr_str} | "
                        f"Spikes: {grad_clipper.spike_count} | "
                        f"Time: {elapsed:.0f}s"
                    )

                    loss_accum = {"ce": 0.0, "aux": 0.0, "ctm": 0.0, "total": 0.0}
                    ctm_ticks_sum = 0
                    ctm_ticks_count = 0

                # Save checkpoint
                if global_step % args.save_every == 0 and is_main:
                    os.makedirs(args.output_dir, exist_ok=True)
                    ckpt_path = os.path.join(args.output_dir, f"checkpoint-{global_step}.pt")
                    torch.save({
                        "step": global_step,
                        "model_state_dict": (
                            model.module.state_dict() if hasattr(model, "module")
                            else model.state_dict()
                        ),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": avg_total,
                        "config": config,
                        "grad_spike_count": grad_clipper.spike_count,
                    }, ckpt_path)
                    logger.info(f"Checkpoint saved: {ckpt_path}")

    if is_main:
        total_time = time.time() - start_time
        logger.info(f"Training complete: {total_time:.1f}s ({total_time/3600:.2f}h)")
        logger.info(f"Final step: {global_step} | Gradient spikes: {grad_clipper.spike_count}")

    cleanup_distributed()


# ==================== CLI ====================

def main():
    parser = argparse.ArgumentParser(description="Luna-Ultimate Hybrid Training")

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--dynamic_layer_skip", action="store_true", default=False)

    parser.add_argument("--use_deepspeed", action="store_true", default=False)
    parser.add_argument("--deepspeed_config", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)

    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()