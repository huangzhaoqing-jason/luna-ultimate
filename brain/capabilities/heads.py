"""Pluggable capability heads — not a single forced paradigm (WM+VLA optional)."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class CapabilitySuite(nn.Module):
    """Language / perception / memory / action / tool heads on shared state."""

    def __init__(self, d_model: int, vocab_size: int = 4096, action_dim: int = 16):
        super().__init__()
        self.language = nn.Linear(d_model, vocab_size)
        self.perception = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        self.memory_write = nn.Linear(d_model, d_model)
        self.memory_read = nn.Linear(d_model, d_model)
        self.action = nn.Linear(d_model, action_dim)
        self.tool = nn.Linear(d_model, 32)
        self.value = nn.Linear(d_model, 1)
        self.register_buffer("memory_slot", torch.zeros(1, d_model), persistent=True)

    def forward(
        self, state: torch.Tensor, enable: Optional[Dict[str, bool]] = None
    ) -> Dict[str, torch.Tensor]:
        enable = enable or {}
        out: Dict[str, torch.Tensor] = {}
        if enable.get("perception", True):
            out["perception"] = self.perception(state)
        if enable.get("memory", True):
            written = torch.tanh(self.memory_write(state))
            # EMA memory update (batch-mean into slot)
            with torch.no_grad():
                self.memory_slot = 0.9 * self.memory_slot + 0.1 * written.mean(0, keepdim=True)
            out["memory"] = self.memory_read(state) + self.memory_slot.to(state.device)
        if enable.get("language", True):
            out["logits"] = self.language(state)
        if enable.get("action", True):
            out["capability_action"] = self.action(state)
        if enable.get("tool", True):
            out["tool_logits"] = self.tool(state)
        if enable.get("value", True):
            out["value"] = self.value(state)
        return out
