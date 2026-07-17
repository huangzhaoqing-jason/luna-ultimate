"""Parallel multi-task slots + multi-instance collective (Pathway 4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn


@dataclass
class GoalSlot:
    goal_id: str
    goal_vec: torch.Tensor
    active: bool = True
    priority: float = 1.0


class MultiTaskRuntime(nn.Module):
    """Maintain N goal slots simultaneously (hyper-brain vs serial chat)."""

    def __init__(self, d_model: int, n_slots: int = 4):
        super().__init__()
        self.n_slots = n_slots
        self.goal_encoder = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        self.slot_gate = nn.Linear(d_model, n_slots)
        self.slots: List[Optional[GoalSlot]] = [None] * n_slots

    def set_goals(self, goal_states: torch.Tensor, ids: Optional[List[str]] = None) -> None:
        """goal_states: [K, D], K <= n_slots."""
        K = min(goal_states.shape[0], self.n_slots)
        encoded = self.goal_encoder(goal_states[:K])
        self.slots = [None] * self.n_slots
        for i in range(K):
            self.slots[i] = GoalSlot(
                goal_id=ids[i] if ids else f"goal_{i}",
                goal_vec=encoded[i].detach(),
                priority=1.0 - 0.05 * i,
            )

    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Mix active goals into state; return per-slot scores."""
        B, D = state.shape
        gate = torch.softmax(self.slot_gate(state), dim=-1)  # [B, n_slots]
        mix = torch.zeros_like(state)
        slot_scores = []
        for i, slot in enumerate(self.slots):
            if slot is None or not slot.active:
                slot_scores.append(torch.zeros(B, device=state.device))
                continue
            g = slot.goal_vec.to(state.device).unsqueeze(0).expand(B, -1)
            contrib = gate[:, i : i + 1] * g * slot.priority
            mix = mix + contrib
            slot_scores.append((state * g).sum(dim=-1))
        scores = torch.stack(slot_scores, dim=-1)
        return {
            "state": state + 0.2 * mix,
            "slot_scores": scores,
            "n_active": torch.tensor(
                sum(1 for s in self.slots if s is not None and s.active),
                device=state.device,
            ),
        }


class CollectivePool:
    """Multiple brain instance handles (digital copy advantage)."""

    def __init__(self):
        self.instances: Dict[str, object] = {}

    def register(self, name: str, brain: object) -> None:
        self.instances[name] = brain

    def names(self) -> List[str]:
        return list(self.instances.keys())

    def broadcast(self, method: str, *args, **kwargs):
        return {n: getattr(b, method)(*args, **kwargs) for n, b in self.instances.items()}
