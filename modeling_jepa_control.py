"""统一 JEPA 控制器：用预测下一状态的 latent 总控全模型。

JEPAController 产出：
  - jepa_ctrl: [B, d] 控制 latent（CTM-JEPA ± V-JEPA 融合）
  - ControlSignals: uncertainty / compute_scale / task_logits / wake_logits

丘脑、CTM ticks、MoE 预算、Meaning/Motor/WA 均消费这些信号。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import LunaConfig
from modeling_ctm import CTM, CTMJEPAPredictor

# 与 TaskType 对齐的任务类别（顺序固定，供 thalamus 映射）
TASK_NAMES: Sequence[str] = (
    "simple", "knowledge", "math", "code", "reasoning", "action", "full",
)

# 可唤醒脑区（与 thalamus wake 名对齐）
WAKE_NAMES: Sequence[str] = (
    "prefrontal", "parietal", "temporal", "hippocampus",
    "cerebellum", "brainstem", "thalamus", "motor",
)


@dataclass
class ControlSignals:
    uncertainty: float          # 0-1，预测误差代理
    compute_scale: float        # 0-1，算力缩放
    task_logits: torch.Tensor   # [B, n_tasks]
    wake_logits: torch.Tensor   # [B, n_wake]
    jepa_ctrl: torch.Tensor     # [B, d]
    next_latent: torch.Tensor   # [B, d]
    jepa_loss: Optional[torch.Tensor] = None


class JEPAController(nn.Module):
    """全模型 JEPA 控制器。

    优先挂钩 CTM 的 jepa_predictor；另有独立轻量头作路由/算力信号。
    """

    def __init__(self, config: LunaConfig, ctm: Optional[CTM] = None):
        super().__init__()
        self.d_model = config.hidden_size
        self.n_neurons = config.ctm_n_neurons
        pred_hidden = min(256, max(32, self.n_neurons // 2))

        # 若 CTM 有 JEPA，复用其 predictor（不注册为子模块，避免 state_dict 重复嵌套）
        if ctm is not None and getattr(ctm, "jepa_predictor", None) is not None:
            object.__setattr__(self, "predictor", ctm.jepa_predictor)
            self._owns_predictor = False
        else:
            self.predictor = CTMJEPAPredictor(
                n_neurons=self.n_neurons, hidden_dim=pred_hidden, predict_horizon=1,
            )
            self._owns_predictor = True

        self.to_neurons = nn.Linear(self.d_model, self.n_neurons, bias=False)
        self.to_ctrl = nn.Linear(self.n_neurons, self.d_model, bias=False)
        self.ctrl_norm = nn.LayerNorm(self.d_model)
        # 视觉融合
        self.vision_fuse = nn.Linear(self.d_model * 2, self.d_model, bias=False)
        # 控制头
        self.task_head = nn.Linear(self.d_model, len(TASK_NAMES), bias=False)
        self.wake_head = nn.Linear(self.d_model, len(WAKE_NAMES), bias=False)
        self.uncert_head = nn.Linear(self.d_model, 1, bias=False)
        self.next_head = nn.Linear(self.d_model, self.d_model, bias=False)

    def encode_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden [B,L,d] → neuron state [B,n]"""
        pooled = hidden.mean(dim=1)
        return self.to_neurons(pooled)

    def from_neuron_state(
        self,
        neuron_state: torch.Tensor,
        vision_latent: Optional[torch.Tensor] = None,
        jepa_loss: Optional[torch.Tensor] = None,
    ) -> ControlSignals:
        """neuron_state [B,n] → ControlSignals"""
        pred = self.predictor(neuron_state)              # [B, n]
        ctrl = self.ctrl_norm(self.to_ctrl(pred))        # [B, d]
        if vision_latent is not None:
            if vision_latent.dim() == 3:
                vision_latent = vision_latent.mean(dim=1)
            ctrl = self.ctrl_norm(self.vision_fuse(torch.cat([ctrl, vision_latent], dim=-1)))

        # 不确定性：预测前后神经元余弦距离（高 = 难）
        with torch.no_grad():
            cos = F.cosine_similarity(pred, neuron_state, dim=-1).mean()
            unc_proxy = float((1.0 - cos).clamp(0, 1).item())
        # 可学习 uncertainty 头（与 proxy 混合）
        unc_learned = torch.sigmoid(self.uncert_head(ctrl)).mean().item()
        uncertainty = 0.5 * unc_proxy + 0.5 * float(unc_learned)
        compute_scale = min(1.0, max(0.1, uncertainty))

        task_logits = self.task_head(ctrl)
        wake_logits = self.wake_head(ctrl)
        next_latent = self.next_head(ctrl)

        return ControlSignals(
            uncertainty=uncertainty,
            compute_scale=compute_scale,
            task_logits=task_logits,
            wake_logits=wake_logits,
            jepa_ctrl=ctrl,
            next_latent=next_latent,
            jepa_loss=jepa_loss,
        )

    def forward(
        self,
        hidden: torch.Tensor,
        vision_latent: Optional[torch.Tensor] = None,
        neuron_state: Optional[torch.Tensor] = None,
        jepa_loss: Optional[torch.Tensor] = None,
    ) -> ControlSignals:
        if neuron_state is None:
            neuron_state = self.encode_hidden(hidden)
        return self.from_neuron_state(neuron_state, vision_latent, jepa_loss)

    def predict_next(self, jepa_ctrl: torch.Tensor) -> torch.Tensor:
        return self.next_head(jepa_ctrl)
