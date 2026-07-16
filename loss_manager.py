"""Loss Manager for Luna-Ultimate: multi-task loss with dynamic weighting.

Manages 4 loss branches:
  1. L_LM:   Language modeling next-token loss (primary)
  2. L_VJEPA: Visual JEPA prediction loss (auxiliary)
  3. L_CTMJ:  CTM-JEPA state prediction loss (auxiliary)
  4. L_MoE:   Expert load balancing loss (structural)

Uses Uncertainty Weighting (Kendall et al., 2018) to dynamically balance
losses by learning per-task homoscedastic uncertainty parameters.
Also supports GradNorm for gradient-based balancing.

Reference:
  - "Multi-Task Learning Using Uncertainty" (Kendall, Gal, Cipolla, 2018)
  - "GradNorm: Gradient Normalization for Adaptive Loss Balancing" (Chen et al., 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math


class UncertaintyWeightedLoss(nn.Module):
    """Multi-task loss with learned uncertainty weighting.

    Each task gets a learned log-variance parameter σ².
    Loss = Σ (L_i / (2σ_i²) + log(σ_i))
    This naturally balances tasks: high-noise tasks get lower weight.

    Args:
        num_tasks: Number of loss branches.
        init_log_var: Initial log variance for each task.
        method: "uncertainty" or "gradnorm".
    """

    def __init__(
        self,
        num_tasks: int = 4,
        init_log_var: float = 0.0,
        method: str = "uncertainty",
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.method = method

        # Learnable log variances: log(σ²)
        self.log_vars = nn.Parameter(torch.ones(num_tasks) * init_log_var)

        # Task names for logging
        self.task_names = ["lm", "vjepa", "ctmj", "moe"]

    def forward(
        self,
        losses: Dict[str, torch.Tensor],
        step: int = 0,
        total_steps: int = 100000,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute weighted multi-task loss.

        Args:
            losses: Dict with keys "lm", "vjepa", "ctmj", "moe".
            step: Current training step (for annealing).
            total_steps: Total training steps.

        Returns:
            total_loss: scalar weighted sum.
            stats: dict of per-task losses and weights.
        """
        stats = {}

        if self.method == "uncertainty":
            total_loss = torch.tensor(0.0, device=self.log_vars.device)
            for i, name in enumerate(self.task_names):
                if name in losses and losses[name] is not None:
                    precision = torch.exp(-self.log_vars[i])
                    weighted = precision * losses[name] + self.log_vars[i] * 0.5
                    total_loss = total_loss + weighted
                    stats[f"{name}_loss"] = losses[name].item()
                    stats[f"{name}_weight"] = precision.item()
                    stats[f"{name}_logvar"] = self.log_vars[i].item()
        else:
            # Simple weighted sum with static weights
            weights = {
                "lm": 1.0,
                "vjepa": 0.1,
                "ctmj": 0.05,
                "moe": 0.01,
            }
            total_loss = torch.tensor(0.0, device=self.log_vars.device)
            for name, weight in weights.items():
                if name in losses and losses[name] is not None:
                    total_loss = total_loss + weight * losses[name]
                    stats[f"{name}_loss"] = losses[name].item()
                    stats[f"{name}_weight"] = weight

        stats["total_loss"] = total_loss.item()
        return total_loss, stats


class GradNormLossBalancer:
    """GradNorm-based adaptive loss balancing.

    Dynamically adjusts task weights so that all tasks learn at similar speeds.
    Weights are updated based on the ratio of each task's gradient norm to
    the average gradient norm.

    Reference: "GradNorm: Gradient Normalization for Adaptive Loss Balancing"
    """

    def __init__(
        self,
        num_tasks: int = 4,
        alpha: float = 0.12,
        lr: float = 0.001,
    ):
        self.num_tasks = num_tasks
        self.alpha = alpha
        self.lr = lr

        # Task weights (initialized to 1.0)
        self.task_weights = torch.ones(num_tasks)
        self.initial_losses = None
        self.task_names = ["lm", "vjepa", "ctmj", "moe"]

    def compute_weights(
        self,
        losses: Dict[str, torch.Tensor],
        model: nn.Module,
        step: int,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """Compute GradNorm-adjusted task weights.

        Args:
            losses: Dict of per-task losses.
            model: The model (for gradient computation).
            step: Current step.

        Returns:
            weighted_losses: Dict of weighted per-task losses.
            stats: Dict of task weights.
        """
        # Record initial losses for relative inverse training rate
        loss_values = []
        for name in self.task_names:
            if name in losses and losses[name] is not None:
                loss_values.append(losses[name].detach())
            else:
                loss_values.append(torch.tensor(0.0))

        if self.initial_losses is None:
            self.initial_losses = torch.stack(loss_values)

        # Compute inverse training rates
        with torch.no_grad():
            current_losses = torch.stack(loss_values)
            inv_rates = current_losses / (self.initial_losses + 1e-8)
            # Target: all tasks should have similar inverse training rates
            mean_inv_rate = inv_rates.mean()

        # Simple approximation: adjust weights based on loss ratios
        # Tasks with higher relative loss get higher weight
        total_loss_sum = sum(max(l.item(), 1e-8) for l in loss_values)
        weights = {}
        for i, name in enumerate(self.task_names):
            if name in losses and losses[name] is not None:
                w = total_loss_sum / (len(loss_values) * max(loss_values[i].item(), 1e-8))
                weights[name] = w
            else:
                weights[name] = 0.0

        weighted_losses = {}
        for name in self.task_names:
            if name in losses and losses[name] is not None:
                weighted_losses[name] = losses[name] * weights[name]

        stats = {f"{name}_gradnorm_weight": weights[name] for name in self.task_names}
        return weighted_losses, stats


class StageAwareLossManager:
    """Stage-aware loss manager for 3-stage training.

    Stage 1: Freeze backbone, train V-JEPA (vision alignment)
      - L_VJEPA: 1.0, L_LM: 0.0, L_CTMJ: 0.0, L_MoE: 0.0

    Stage 2: Freeze V-JEPA, train backbone + CTM-JEPA (reasoning enhancement)
      - L_LM: 1.0, L_CTMJ: 0.1, L_VJEPA: 0.0, L_MoE: 0.01

    Stage 3: Full joint training (all modules)
      - L_LM: 1.0, L_VJEPA: 0.05, L_CTMJ: 0.05, L_MoE: 0.01

    Args:
        stage: 1, 2, or 3.
        use_uncertainty: Whether to use uncertainty weighting within stages.
    """

    def __init__(self, stage: int = 1, use_uncertainty: bool = True):
        self.stage = stage
        self.use_uncertainty = use_uncertainty

        # Stage-specific static weights (Luna Evolve)
        # Stage 1: text LM pretrain; Stage 2: +CTM-JEPA; Stage 3: multimodal
        self.stage_weights = {
            1: {"lm": 1.0, "vjepa": 0.0, "ctmj": 0.0, "moe": 0.01},
            2: {"lm": 1.0, "vjepa": 0.0, "ctmj": 0.1, "moe": 0.01},
            3: {"lm": 1.0, "vjepa": 0.05, "ctmj": 0.05, "moe": 0.01},
        }

        if use_uncertainty:
            self.uncertainty_module = UncertaintyWeightedLoss(num_tasks=4, method="uncertainty")
        else:
            self.uncertainty_module = None

        self.gradnorm = GradNormLossBalancer(num_tasks=4)

    def get_stage_weights(self) -> Dict[str, float]:
        """Get static weights for current stage."""
        return self.stage_weights.get(self.stage, self.stage_weights[3])

    def compute_loss(
        self,
        losses: Dict[str, torch.Tensor],
        model: Optional[nn.Module] = None,
        step: int = 0,
        total_steps: int = 100000,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute stage-aware weighted loss.

        Args:
            losses: Dict with keys "lm", "vjepa", "ctmj", "moe".
            model: Model for GradNorm (optional).
            step: Current step.
            total_steps: Total steps.

        Returns:
            total_loss: scalar.
            stats: dict of per-task stats.
        """
        if self.uncertainty_module is not None:
            return self.uncertainty_module(losses, step, total_steps)

        # Fallback: static weights
        weights = self.get_stage_weights()
        device = None
        for v in losses.values():
            if torch.is_tensor(v):
                device = v.device
                break
        total_loss = torch.zeros((), device=device) if device is not None else torch.tensor(0.0)
        stats = {}

        for name, weight in weights.items():
            if name in losses and losses[name] is not None and weight > 0:
                total_loss = total_loss + weight * losses[name]
                stats[f"{name}_loss"] = float(losses[name].detach().item())
                stats[f"{name}_weight"] = weight

        stats["total_loss"] = float(total_loss.detach().item())
        stats["stage"] = float(self.stage)
        return total_loss, stats

    def set_stage(self, stage: int):
        """Switch training stage."""
        assert stage in (1, 2, 3), f"Invalid stage: {stage}"
        self.stage = stage

    def get_frozen_modules(self) -> Dict[int, str]:
        """Get which modules to freeze at each stage.

        Returns:
            Dict mapping stage → module name pattern to freeze.
        """
        return {
            1: "vjepa",            # Stage 1: freeze V-JEPA, train text LM
            2: "vjepa",            # Stage 2: freeze V-JEPA, train + CTM-JEPA
            3: "",                 # Stage 3: train everything
        }

    def get_trainable_pattern(self) -> str:
        """Get pattern for modules that should be trainable at current stage."""
        if self.stage == 1:
            return "embed,mamba,mla,moe,lm_head"
        elif self.stage == 2:
            return "ctm,mla,mamba,moe,embed,lm_head"
        else:
            return ""