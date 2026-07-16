"""World-Action（WA）：世界模型预测下一状态 + 建议动作。

观测序列 → JEPA latent → predict_next + optional action prior。
与 VLA 配套：WA 负责「世界会怎样」，VLA 负责「怎么动」。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import LunaConfig
from modeling_ctm import CTMJEPAPredictor


@dataclass
class WorldActionOutput:
    current_latent: torch.Tensor       # [B, d]
    next_latent: torch.Tensor          # [B, d]
    action_prior_logits: torch.Tensor  # [B, n_actions]
    world_loss: Optional[torch.Tensor] = None


class WorldActionModule(nn.Module):
    """观测 hidden → 当前/下一世界状态 + 动作先验。"""

    def __init__(self, config: LunaConfig, n_actions: int = 10):
        super().__init__()
        self.d_model = config.hidden_size
        self.n_actions = n_actions
        # 观测池化 → 世界 latent
        self.to_latent = nn.Sequential(
            nn.Linear(self.d_model, self.d_model, bias=False),
            nn.LayerNorm(self.d_model),
        )
        # 复用 CTM-JEPA 风格预测器：current → next（在 d_model 空间）
        pred_hidden = min(256, max(32, self.d_model // 2))
        # CTMJEPAPredictor 期望 n_neurons 维；这里把 d_model 当「神经元」维用
        self.predictor = CTMJEPAPredictor(
            n_neurons=self.d_model,
            hidden_dim=pred_hidden,
            predict_horizon=1,
        )
        self.action_prior = nn.Linear(self.d_model, n_actions, bias=False)

    def encode_obs(self, obs_hidden: torch.Tensor) -> torch.Tensor:
        """obs_hidden: [B, L, d] → [B, d]"""
        return self.to_latent(obs_hidden.mean(dim=1))

    def forward(
        self,
        obs_hidden: torch.Tensor,
        next_obs_hidden: Optional[torch.Tensor] = None,
    ) -> WorldActionOutput:
        current = self.encode_obs(obs_hidden)
        next_pred = self.predictor(current)
        prior = self.action_prior(current)

        world_loss = None
        if next_obs_hidden is not None:
            target = self.encode_obs(next_obs_hidden).detach()
            world_loss = 1.0 - F.cosine_similarity(next_pred, target, dim=-1).mean()

        return WorldActionOutput(
            current_latent=current,
            next_latent=next_pred,
            action_prior_logits=prior,
            world_loss=world_loss,
        )

    def suggest_action(self, obs_hidden: torch.Tensor) -> Dict[str, object]:
        out = self.forward(obs_hidden)
        idx = int(out.action_prior_logits[0].argmax().item())
        return {
            "action_id": idx,
            "prior_logits": out.action_prior_logits,
            "next_latent": out.next_latent,
            "current_latent": out.current_latent,
        }
