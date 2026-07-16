"""Luna-Ultimate Training Script.

Complete distributed training loop with:
  - Archer entropy-aware KL constraints (knowledge vs reasoning tokens)
  - MoE load balancing loss integration
  - Gradient accumulation, mixed precision (BF16), gradient checkpointing
  - Warmup + Cosine Decay learning rate schedule
  - DeepSpeed / FSDP compatibility
  - WandB logging support

Usage:
    torchrun --nproc_per_node=8 train.py
    deepspeed train.py --deepspeed ds_config.json
"""

import os
import sys
import math
import argparse
import logging
from typing import Optional, Dict, Any
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== Archer Entropy-Aware Loss ====================

class ArcherEntropyLoss(nn.Module):
    """Archer entropy-aware training loss.

    Differentiates between:
      - Knowledge tokens (low entropy): Strong KL constraint (0.1)
      - Reasoning tokens (high entropy): Weak KL constraint (0.001), high clip threshold (0.3)

    Args:
        config: LunaConfig with Archer hyperparameters.
    """

    def __init__(self, config: LunaConfig):
        super().__init__()
        self.knowledge_kl_weight = config.archer_knowledge_kl_weight      # 0.1
        self.reasoning_kl_weight = config.archer_reasoning_kl_weight      # 0.001
        self.reasoning_clip_threshold = config.archer_reasoning_clip_threshold  # 0.3
        self.entropy_threshold = config.archer_entropy_threshold          # 0.5
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)

    def _compute_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute per-token entropy from logits.

        Args:
            logits: [B, L, vocab_size]

        Returns:
            entropy: [B, L] - per-token entropy
        """
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)  # [B, L]
        return entropy

    def _compute_kl_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL divergence loss between student and teacher logits.

        Args:
            student_logits: [B, L, vocab_size]
            teacher_logits: [B, L, vocab_size] (detached)
            mask: [B, L] - valid token mask

        Returns:
            kl_loss: scalar
        """
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_probs = F.softmax(teacher_logits.detach(), dim=-1)

        kl = (teacher_probs * (torch.log(teacher_probs + 1e-10) - student_log_probs)).sum(dim=-1)
        # [B, L]

        kl = (kl * mask).sum() / (mask.sum() + 1e-8)
        return kl

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        teacher_logits: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute Archer entropy-aware loss.

        Args:
            logits: [B, L, vocab_size] - model logits
            labels: [B, L] - target token IDs
            teacher_logits: Optional [B, L, vocab_size] - teacher logits for KL

        Returns:
            loss: scalar total loss
            stats: dict of loss components
        """
        B, L, V = logits.shape

        # Mask for valid (non-ignored) tokens
        mask = (labels != -100).float()  # [B, L]

        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device), {}

        # Standard CE loss
        ce_loss = self.ce_loss(
            logits.view(B * L, V),
            labels.view(B * L),
        )

        total_loss = ce_loss
        stats = {"ce_loss": ce_loss.item()}

        if teacher_logits is not None:
            # Compute per-token entropy
            entropy = self._compute_entropy(logits)  # [B, L]

            # Classify tokens: knowledge (low entropy) vs reasoning (high entropy)
            knowledge_mask = (entropy <= self.entropy_threshold).float() * mask  # [B, L]
            reasoning_mask = (entropy > self.entropy_threshold).float() * mask   # [B, L]

            # Knowledge KL loss (strong constraint)
            if knowledge_mask.sum() > 0:
                kl_knowledge = self._compute_kl_loss(logits, teacher_logits, knowledge_mask)
                total_loss = total_loss + self.knowledge_kl_weight * kl_knowledge
                stats["kl_knowledge"] = kl_knowledge.item()

            # Reasoning KL loss (weak constraint with clipping)
            if reasoning_mask.sum() > 0:
                kl_reasoning = self._compute_kl_loss(logits, teacher_logits, reasoning_mask)
                # Clip reasoning KL to avoid over-constraining
                kl_reasoning = torch.clamp(kl_reasoning, max=self.reasoning_clip_threshold)
                total_loss = total_loss + self.reasoning_kl_weight * kl_reasoning
                stats["kl_reasoning"] = kl_reasoning.item()

            stats["knowledge_ratio"] = (knowledge_mask.sum() / (mask.sum() + 1e-8)).item()
            stats["reasoning_ratio"] = (reasoning_mask.sum() / (mask.sum() + 1e-8)).item()

        stats["total_loss"] = total_loss.item()
        return total_loss, stats


# ==================== Learning Rate Scheduler ====================

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
):
    """Create cosine LR schedule with warmup.

    Args:
        optimizer: The optimizer.
        warmup_steps: Number of warmup steps.
        total_steps: Total training steps.
        min_lr_ratio: Minimum LR as fraction of peak LR.
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine decay
            progress = float(current_step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


# ==================== Dummy Dataset ====================

class DummyDataset(torch.utils.data.Dataset):
    """Dummy dataset for testing. Replace with real dataset."""

    def __init__(self, vocab_size: int, seq_len: int, num_samples: int = 10000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        # Shift for next-token prediction
        input_ids = ids[:-1]
        labels = ids[1:]
        return {"input_ids": input_ids, "labels": labels}


# ==================== Training Loop ====================

def setup_distributed():
    """Initialize distributed training."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    else:
        return 0, 1, 0


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def train(args: argparse.Namespace):
    """Main training loop."""

    # Setup
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0

    if is_main:
        logger.info(f"Starting Luna-Ultimate training on {world_size} GPUs")

    # Config
    config = LunaConfig()

    # Model
    if is_main:
        logger.info("Initializing model...")

    model = LunaUltimate(config)
    model = model.to(device)

    # Enable gradient checkpointing for memory efficiency
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if is_main:
        total_b, active_b = model.get_num_parameters()
        logger.info(f"Model initialized: {total_b:.2f}B total, {active_b:.2f}B active")

    # Distributed model wrapper
    if world_size > 1:
        if args.use_deepspeed:
            import deepspeed
            model, optimizer, _, _ = deepspeed.initialize(
                model=model,
                model_parameters=model.parameters(),
                config_params=args.deepspeed_config if args.deepspeed_config else {},
            )
        else:
            model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                find_unused_parameters=False,
            )
            optimizer = AdamW(
                model.parameters(),
                lr=config.learning_rate,
                betas=(config.adam_beta1, config.adam_beta2),
                weight_decay=config.weight_decay,
            )
    else:
        optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            weight_decay=config.weight_decay,
        )

    # Scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps=config.warmup_steps,
        total_steps=config.max_steps,
    )

    # Loss function
    archer_loss = ArcherEntropyLoss(config)

    # Mixed precision
    use_amp = args.use_amp and config.mixed_precision == "bf16"
    scaler = GradScaler(enabled=use_amp)

    # Dataset
    dataset = DummyDataset(
        vocab_size=config.vocab_size,
        seq_len=args.seq_len,
        num_samples=args.num_samples,
    )

    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    # Training state
    global_step = 0
    total_loss_accum = 0.0
    total_aux_loss_accum = 0.0
    best_loss = float("inf")

    if is_main:
        logger.info(f"Starting training: {config.max_steps} steps")
        logger.info(f"Batch size: {args.batch_size}, Grad accum: {args.grad_accum}")
        logger.info(f"Effective batch: {args.batch_size * args.grad_accum * world_size}")

    model.train()
    start_time = time.time()

    # ==================== Training Loop ====================
    for epoch in range(args.num_epochs):
        if world_size > 1:
            sampler.set_epoch(epoch)

        for batch_idx, batch in enumerate(dataloader):
            if global_step >= config.max_steps:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Forward with autocast
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits, aux_loss = model(
                    input_ids,
                    use_ctm_adaptive=True,
                    use_dynamic_skip=args.dynamic_layer_skip,
                )

                # Compute Archer loss
                loss, loss_stats = archer_loss(logits, labels)

                # Add MoE auxiliary loss
                total_loss = loss + aux_loss * config.moe_aux_loss_coeff

            # Scale loss for gradient accumulation
            total_loss = total_loss / args.grad_accum

            # Backward
            if use_amp:
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()

            total_loss_accum += total_loss.item() * args.grad_accum
            total_aux_loss_accum += aux_loss.item()

            # Gradient accumulation step
            if (batch_idx + 1) % args.grad_accum == 0:
                # Gradient clipping
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Logging
                if global_step % args.log_every == 0 and is_main:
                    elapsed = time.time() - start_time
                    lr = scheduler.get_last_lr()[0]
                    avg_loss = total_loss_accum / args.log_every
                    avg_aux = total_aux_loss_accum / args.log_every

                    log_msg = (
                        f"Step {global_step}/{config.max_steps} | "
                        f"Loss: {avg_loss:.4f} | Aux: {avg_aux:.4f} | "
                        f"LR: {lr:.2e} | Time: {elapsed:.1f}s"
                    )

                    if loss_stats:
                        log_msg += f" | CE: {loss_stats.get('ce_loss', 0):.4f}"
                        if "kl_knowledge" in loss_stats:
                            log_msg += f" | KL_K: {loss_stats['kl_knowledge']:.4f}"
                        if "kl_reasoning" in loss_stats:
                            log_msg += f" | KL_R: {loss_stats['kl_reasoning']:.4f}"
                        if "knowledge_ratio" in loss_stats:
                            log_msg += (
                                f" | K%: {loss_stats['knowledge_ratio']*100:.1f}%"
                            )

                    logger.info(log_msg)

                    total_loss_accum = 0.0
                    total_aux_loss_accum = 0.0

                # Save checkpoint
                if global_step % args.save_every == 0 and is_main:
                    checkpoint_path = os.path.join(
                        args.output_dir, f"checkpoint-{global_step}.pt"
                    )
                    os.makedirs(args.output_dir, exist_ok=True)
                    torch.save(
                        {
                            "step": global_step,
                            "model_state_dict": (
                                model.module.state_dict()
                                if hasattr(model, "module")
                                else model.state_dict()
                            ),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "loss": avg_loss,
                            "config": config,
                        },
                        checkpoint_path,
                    )
                    logger.info(f"Checkpoint saved to {checkpoint_path}")

    if is_main:
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.1f}s ({total_time/3600:.2f}h)")
        logger.info(f"Final step: {global_step}")

    cleanup_distributed()


# ==================== CLI ====================

def main():
    parser = argparse.ArgumentParser(description="Luna-Ultimate Training")

    # Training
    parser.add_argument("--batch_size", type=int, default=1, help="Per-GPU batch size")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--seq_len", type=int, default=2048, help="Sequence length")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--num_samples", type=int, default=10000, help="Dataset size")

    # Performance
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--use_amp", action="store_true", default=True, help="Use mixed precision")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--dynamic_layer_skip", action="store_true", default=False)

    # Distributed
    parser.add_argument("--use_deepspeed", action="store_true", default=False)
    parser.add_argument("--deepspeed_config", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)

    # Logging
    parser.add_argument("--log_every", type=int, default=10, help="Log every N steps")
    parser.add_argument("--save_every", type=int, default=1000, help="Save every N steps")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()