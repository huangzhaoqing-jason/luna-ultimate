"""Luna Evolve training — LunaUltimateFused with presets and smoke mode.

Usage:
    python train.py --preset tiny --smoke --batch_size 2 --seq_len 64 --max_steps 5
    torchrun --nproc_per_node=8 train.py --preset 7b
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler

from config import LunaConfig
from modeling_luna_ultimate import LunaUltimateFused

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def create_hybrid_optimizer(model: nn.Module, config: LunaConfig) -> torch.optim.Optimizer:
    mamba_ssm_params, mamba_conv_params, mamba_proj_params = [], [], []
    mla_attn_params, moe_router_params, moe_expert_params = [], [], []
    ctm_params, other_params = [], []

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
        elif "moe" in name or "router" in name or "expert" in name:
            if "router" in name:
                moe_router_params.append(param)
            else:
                moe_expert_params.append(param)
        elif "ctm" in name:
            ctm_params.append(param)
        else:
            other_params.append(param)

    base_lr = config.learning_rate
    param_groups = [
        {"params": mamba_ssm_params, "lr": base_lr * 0.1, "name": "mamba_ssm"},
        {"params": mamba_conv_params, "lr": base_lr * 0.5, "name": "mamba_conv"},
        {"params": mamba_proj_params, "lr": base_lr * 0.8, "name": "mamba_proj"},
        {"params": mla_attn_params, "lr": base_lr, "name": "mla_attn"},
        {"params": moe_router_params, "lr": base_lr * 1.5, "name": "moe_router"},
        {"params": moe_expert_params, "lr": base_lr, "name": "moe_expert"},
        {"params": ctm_params, "lr": base_lr, "name": "ctm"},
        {"params": other_params, "lr": base_lr, "name": "other"},
    ]
    param_groups = [g for g in param_groups if len(g["params"]) > 0]
    return AdamW(
        param_groups,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.weight_decay,
    )


class ArcherEntropyLoss(nn.Module):
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

    def _compute_kl_loss(self, student_logits, teacher_logits, mask):
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_probs = F.softmax(teacher_logits.detach(), dim=-1)
        kl = (teacher_probs * (torch.log(teacher_probs + 1e-10) - student_log_probs)).sum(dim=-1)
        return (kl * mask).sum() / (mask.sum() + 1e-8)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        teacher_logits: Optional[torch.Tensor] = None,
        ctm_ticks: Optional[float] = None,
        global_step: int = 0,
        total_steps: int = 100000,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        B, L, V = logits.shape
        mask = (labels != -100).float()
        if mask.sum() == 0:
            return torch.zeros((), device=logits.device), {}

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

        if ctm_ticks is not None and ctm_ticks > 0:
            ctm_weight = 0.01 * (1.0 - global_step / max(1, total_steps)) + 0.001
            ctm_loss = ctm_weight * (ctm_ticks / 4.0)
            total_loss = total_loss + ctm_loss
            stats["ctm_loss"] = float(ctm_loss) if not torch.is_tensor(ctm_loss) else ctm_loss.item()
            stats["ctm_ticks"] = float(ctm_ticks)

        stats["total_loss"] = total_loss.item()
        return total_loss, stats


class AdaptiveGradientClipper:
    def __init__(self, max_grad_norm: float = 1.0, history_size: int = 100, spike_threshold_multiplier: float = 5.0):
        self.max_grad_norm = max_grad_norm
        self.history = deque(maxlen=history_size)
        self.spike_threshold_multiplier = spike_threshold_multiplier
        self.spike_count = 0

    def clip(self, parameters) -> float:
        total_norm = torch.nn.utils.clip_grad_norm_(parameters, self.max_grad_norm)
        val = total_norm.item() if torch.is_tensor(total_norm) else float(total_norm)
        self.history.append(val)
        if len(self.history) >= 10:
            avg_norm = sum(self.history) / len(self.history)
            if val > avg_norm * self.spike_threshold_multiplier:
                self.spike_count += 1
                torch.nn.utils.clip_grad_norm_(parameters, self.max_grad_norm * 0.5)
        return val


class HybridLRScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, ssm_warmup_ratio=0.1, cooldown_start_ratio=0.8, min_lr_ratio=0.01):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.ssm_warmup_end = int(total_steps * ssm_warmup_ratio)
        self.cooldown_start = int(total_steps * cooldown_start_ratio)
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def get_lr(self, step: int) -> List[float]:
        lrs = []
        for i, group in enumerate(self.optimizer.param_groups):
            base_lr = self.base_lrs[i]
            group_name = group.get("name", "other")
            if step < self.warmup_steps:
                lr = base_lr * (step / max(1, self.warmup_steps))
            elif step < self.ssm_warmup_end:
                if "mamba_ssm" in group_name:
                    progress = (step - self.warmup_steps) / max(1, self.ssm_warmup_end - self.warmup_steps)
                    lr = base_lr * (0.1 + 0.9 * progress)
                else:
                    progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
                    lr = base_lr * max(self.min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
            elif step >= self.cooldown_start:
                progress = (step - self.cooldown_start) / max(1, self.total_steps - self.cooldown_start)
                if "moe_router" in group_name:
                    lr = base_lr * max(self.min_lr_ratio * 0.1, 1.0 - progress)
                else:
                    lr = base_lr * max(self.min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
            else:
                progress = (step - self.ssm_warmup_end) / max(1, self.cooldown_start - self.ssm_warmup_end)
                lr = base_lr * max(self.min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
            lrs.append(lr)
        return lrs

    def step(self, step: int):
        for group, lr in zip(self.optimizer.param_groups, self.get_lr(step)):
            group["lr"] = lr


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, vocab_size: int, seq_len: int, num_samples: int = 10000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        return {"input_ids": ids[:-1], "labels": ids[1:]}


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def train(args: argparse.Namespace):
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0

    config = LunaConfig.from_preset(args.preset)
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate

    if args.smoke:
        args.seq_len = min(args.seq_len, 64)
        args.num_samples = min(args.num_samples, 32)
        args.num_workers = 0
        config.max_steps = min(config.max_steps, args.max_steps or 5)

    if is_main:
        logger.info("=" * 60)
        logger.info("  Luna Evolve Training (LunaUltimateFused)")
        logger.info(f"  Preset: {config.preset_name} | GPUs: {world_size} | Device: {device}")
        logger.info("=" * 60)

    model = LunaUltimateFused(config).to(device)
    model.set_training_stage(args.stage)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if is_main:
        total_b, active_b = model.get_num_parameters()
        logger.info(f"Model: {total_b:.4f}B total | {active_b:.4f}B active")

    use_hybrid_schedule = True
    if world_size > 1 and args.use_deepspeed:
        import deepspeed
        model, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=[p for p in model.parameters() if p.requires_grad],
            config_params=args.deepspeed_config if args.deepspeed_config else {},
        )
        lr_scheduler = None
        use_hybrid_schedule = False
    else:
        if world_size > 1:
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank] if device.type == "cuda" else None,
                find_unused_parameters=True,
            )
        raw = model.module if hasattr(model, "module") else model
        optimizer = create_hybrid_optimizer(raw, config)
        lr_scheduler = HybridLRScheduler(
            optimizer, warmup_steps=config.warmup_steps, total_steps=config.max_steps
        )

    archer_loss = ArcherEntropyLoss(config)
    grad_clipper = AdaptiveGradientClipper(max_grad_norm=config.max_grad_norm)
    use_amp = args.use_amp and device.type == "cuda" and config.mixed_precision == "bf16"
    scaler = GradScaler(enabled=use_amp)

    dataset = DummyDataset(config.vocab_size, args.seq_len, args.num_samples)
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, sampler=sampler,
            num_workers=args.num_workers, pin_memory=device.type == "cuda",
        )
    else:
        sampler = None
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=device.type == "cuda",
        )

    global_step = 0
    loss_accum = {"ce": 0.0, "aux": 0.0, "ctm": 0.0, "total": 0.0}
    start_time = time.time()
    ctm_ticks_sum = 0.0
    ctm_ticks_count = 0

    if is_main:
        logger.info(
            f"Steps: {config.max_steps} | Batch: {args.batch_size} | "
            f"GradAccum: {args.grad_accum} | Stage: {args.stage}"
        )

    model.train()
    for epoch in range(args.num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch_idx, batch in enumerate(dataloader):
            if global_step >= config.max_steps:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            raw_model = model.module if hasattr(model, "module") else model

            amp_ctx = (
                autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp)
                if device.type == "cuda"
                else torch.enable_grad()
            )
            with amp_ctx:
                outputs = raw_model(
                    input_ids,
                    use_ctm_adaptive=True,
                    use_dynamic_skip=args.dynamic_layer_skip,
                    return_all_losses=True,
                )
                avg_ticks = float(outputs.get("avg_ticks", torch.tensor(raw_model.last_avg_ticks)).item())
                ctm_ticks_sum += avg_ticks
                ctm_ticks_count += 1

                text_logits = outputs["lm_logits"]
                # Align label length
                if labels.shape[1] != text_logits.shape[1]:
                    m = min(labels.shape[1], text_logits.shape[1])
                    text_logits = text_logits[:, :m, :]
                    labels = labels[:, :m]

                loss, loss_stats = archer_loss(
                    text_logits, labels,
                    ctm_ticks=avg_ticks,
                    global_step=global_step,
                    total_steps=config.max_steps,
                )
                moe_loss = outputs.get("moe_loss", torch.zeros((), device=device))
                ctmj = outputs.get("ctmj_loss", torch.zeros((), device=device))
                total_loss = loss + moe_loss + 0.1 * ctmj
                total_loss = total_loss / args.grad_accum

            if use_amp:
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()

            loss_accum["ce"] += loss_stats.get("ce_loss", 0)
            loss_accum["aux"] += float(moe_loss.detach().item())
            loss_accum["ctm"] += loss_stats.get("ctm_loss", 0)
            loss_accum["total"] += float(total_loss.detach().item()) * args.grad_accum

            if (batch_idx + 1) % args.grad_accum == 0:
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
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.log_every == 0 and is_main:
                    elapsed = time.time() - start_time
                    avg_ticks_val = ctm_ticks_sum / max(1, ctm_ticks_count)
                    logger.info(
                        f"Step {global_step:>6d}/{config.max_steps} | "
                        f"Loss: {loss_accum['total']/args.log_every:.4f} | "
                        f"CE: {loss_accum['ce']/args.log_every:.4f} | "
                        f"Aux: {loss_accum['aux']/args.log_every:.4f} | "
                        f"CTM: {loss_accum['ctm']/args.log_every:.4f} | "
                        f"GradNorm: {grad_norm:.2f} | Ticks: {avg_ticks_val:.2f} | "
                        f"Time: {elapsed:.0f}s"
                    )
                    loss_accum = {"ce": 0.0, "aux": 0.0, "ctm": 0.0, "total": 0.0}
                    ctm_ticks_sum = 0.0
                    ctm_ticks_count = 0

                if global_step % args.save_every == 0 and is_main:
                    os.makedirs(args.output_dir, exist_ok=True)
                    ckpt_path = os.path.join(args.output_dir, f"checkpoint-{global_step}.pt")
                    torch.save({
                        "step": global_step,
                        "model_state_dict": raw_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": config.to_dict(),
                        "preset": config.preset_name,
                    }, ckpt_path)
                    logger.info(f"Checkpoint saved: {ckpt_path}")

    if is_main:
        logger.info(f"Training complete at step {global_step}")
        if args.smoke:
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save({
                "step": global_step,
                "model_state_dict": (model.module if hasattr(model, "module") else model).state_dict(),
                "config": config.to_dict(),
                "preset": config.preset_name,
            }, os.path.join(args.output_dir, "smoke_final.pt"))

    cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(description="Luna Evolve Training")
    parser.add_argument("--preset", type=str, default="tiny")
    parser.add_argument("--smoke", action="store_true", help="Tiny smoke run")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--use_amp", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--dynamic_layer_skip", action="store_true", default=False)
    parser.add_argument("--use_deepspeed", action="store_true", default=False)
    parser.add_argument("--deepspeed_config", type=str, default=None)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
