"""丘脑动态路由（GWT）：按任务类型按需唤醒脑区。

简单任务只唤醒小脑+少量专家（~10% 算力）；中等任务唤醒颞叶+前额叶（~30%）；
顶级复杂推理才全脑激活。这是「低算力高性能」的核心保障。
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import LunaConfig


class TaskType(enum.Enum):
    SIMPLE = "simple"          # 常规补全/闲聊 → 小脑
    KNOWLEDGE = "knowledge"    # 事实/检索 → 颞叶
    MATH = "math"              # 数学/符号 → 顶叶
    CODE = "code"              # 代码 → 前额叶+小脑+颞叶
    REASONING = "reasoning"    # 复杂推理 → 前额叶+顶叶+颞叶
    FULL = "full"              # 全脑


@dataclass
class RoutePlan:
    task_type: TaskType
    wake: Set[str]             # 脑区名集合
    expert_budget: float       # 0-1，专家激活比例预算
    use_cerebellum_cache: bool = False
    use_parietal: bool = False
    use_rag: bool = False
    reason: str = ""


# 关键词启发式路由（轻量，CPU 友好；生产可换分类器）
_MATH_PAT = re.compile(r"\b(math|方程|积分|证明|theorem|prove|geometry|代数|微积分)\b", re.I)
_CODE_PAT = re.compile(r"\b(code|python|function|bug|算法|编程|script|refactor|API)\b", re.I)
_REASON_PAT = re.compile(r"\b(为什么|why|how to|plan|推理|证明|设计|架构|strategy)\b", re.I)
_KNOW_PAT = re.compile(r"\b(什么是|what is|who|when|事实|定义|explain)\b", re.I)


def classify_task(text: str) -> TaskType:
    if not text or not text.strip():
        return TaskType.SIMPLE
    if _MATH_PAT.search(text):
        return TaskType.MATH
    if _CODE_PAT.search(text):
        return TaskType.CODE
    if _REASON_PAT.search(text):
        return TaskType.REASONING
    if _KNOW_PAT.search(text):
        return TaskType.KNOWLEDGE
    return TaskType.SIMPLE


# 任务类型 → 唤醒脑区
_WAKE_MAP: Dict[TaskType, Set[str]] = {
    TaskType.SIMPLE: {"cerebellum", "thalamus"},
    TaskType.KNOWLEDGE: {"temporal", "thalamus", "rag"},
    TaskType.MATH: {"parietal", "prefrontal", "thalamus"},
    TaskType.CODE: {"prefrontal", "cerebellum", "temporal", "thalamus"},
    TaskType.REASONING: {"prefrontal", "parietal", "temporal", "thalamus"},
    TaskType.FULL: {
        "prefrontal", "parietal", "temporal", "cerebellum",
        "hippocampus", "thalamus", "rag",
    },
}

_BUDGET_MAP: Dict[TaskType, float] = {
    TaskType.SIMPLE: 0.10,
    TaskType.KNOWLEDGE: 0.25,
    TaskType.MATH: 0.35,
    TaskType.CODE: 0.40,
    TaskType.REASONING: 0.55,
    TaskType.FULL: 1.0,
}


class ThalamusRouter(nn.Module):
    """丘脑路由器：文本 → RoutePlan；可选学到的路由头（训练时用）。"""

    def __init__(self, config: LunaConfig):
        super().__init__()
        self.config = config
        self.d_model = config.hidden_size
        # 可学习路由头（可选，默认用启发式；训练时可用）
        n_types = len(TaskType)
        self.route_head = nn.Linear(self.d_model, n_types, bias=False)
        self._type_list = list(TaskType)

    def plan_from_text(self, text: str) -> RoutePlan:
        t = classify_task(text)
        wake = set(_WAKE_MAP[t])
        return RoutePlan(
            task_type=t,
            wake=wake,
            expert_budget=_BUDGET_MAP[t],
            use_cerebellum_cache=(t == TaskType.SIMPLE),
            use_parietal=(t in (TaskType.MATH, TaskType.REASONING, TaskType.FULL)),
            use_rag=(t in (TaskType.KNOWLEDGE, TaskType.CODE, TaskType.FULL)),
            reason=f"heuristic:{t.value}",
        )

    def plan_from_hidden(self, prompt_hidden: torch.Tensor) -> RoutePlan:
        """用学到的路由头（训练稳定性对照）；推理默认用 plan_from_text。"""
        pooled = prompt_hidden.mean(dim=1)  # [B, d]
        logits = self.route_head(pooled)    # [B, n_types]
        idx = int(logits[0].argmax().item())
        t = self._type_list[idx]
        wake = set(_WAKE_MAP[t])
        return RoutePlan(
            task_type=t,
            wake=wake,
            expert_budget=_BUDGET_MAP[t],
            use_cerebellum_cache=(t == TaskType.SIMPLE),
            use_parietal=(t in (TaskType.MATH, TaskType.REASONING, TaskType.FULL)),
            use_rag=(t in (TaskType.KNOWLEDGE, TaskType.CODE, TaskType.FULL)),
            reason=f"learned:{t.value}",
        )

    def expert_top_k(self, config: LunaConfig, plan: RoutePlan) -> int:
        """按预算缩放 top_k，简单任务少激活专家。"""
        base = config.num_expert_activated
        k = max(1, int(round(base * plan.expert_budget * 2)))  # 粗缩放
        return min(k, config.num_routed_experts)
