"""丘脑动态路由（GWT）：按任务类型按需唤醒脑区 + 真调度开关。

RoutePlan 现在携带可被 forward 真消费的开关：
  - ctm_ticks：CTM 自适应 tick 上限（简单任务 1，复杂 4）
  - enable_layer_skip：是否允许动态跳层
  - enable_rag / enable_parietal / enable_hippo / enable_cerebellum / enable_action
  - cache_lookup：小脑缓存命中即跳主干（最省算力路径）

简单任务 ~10% 算力；中等 ~30%；顶级复杂才全脑。
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import LunaConfig
from modeling_neuroarch import SubRegion, subregions_for_task


class TaskType(enum.Enum):
    SIMPLE = "simple"
    KNOWLEDGE = "knowledge"
    MATH = "math"
    CODE = "code"
    REASONING = "reasoning"
    ACTION = "action"        # 需要落地执行（M1/premotor/SMA）
    FULL = "full"


@dataclass
class RoutePlan:
    task_type: TaskType
    wake: Set[str]
    sub_regions: List[str] = field(default_factory=list)
    expert_budget: float = 0.25
    # —— 真调度开关 ——
    ctm_ticks: int = 2
    enable_layer_skip: bool = False
    enable_rag: bool = False
    enable_parietal: bool = False
    enable_hippo: bool = False
    enable_cerebellum: bool = True
    enable_action: bool = False
    cache_lookup: bool = True   # 小脑缓存优先
    reason: str = ""


_MATH_PAT = re.compile(r"\b(math|方程|积分|证明|theorem|prove|geometry|代数|微积分)\b", re.I)
_CODE_PAT = re.compile(r"\b(code|python|function|bug|算法|编程|script|refactor|API|编译|build)\b", re.I)
_REASON_PAT = re.compile(r"\b(为什么|why|how to|plan|推理|证明|设计|架构|strategy)\b", re.I)
_KNOW_PAT = re.compile(r"\b(什么是|what is|who|when|事实|定义|explain)\b", re.I)
_ACTION_PAT = re.compile(r"\b(执行|run|提交|commit|push|clone|部署|deploy|操作|终端|terminal|git)\b", re.I)


def classify_task(text: str) -> TaskType:
    if not text or not text.strip():
        return TaskType.SIMPLE
    if _ACTION_PAT.search(text) and (_CODE_PAT.search(text) or "git" in text.lower()):
        return TaskType.ACTION
    if _MATH_PAT.search(text):
        return TaskType.MATH
    if _CODE_PAT.search(text):
        return TaskType.CODE
    if _REASON_PAT.search(text):
        return TaskType.REASONING
    if _KNOW_PAT.search(text):
        return TaskType.KNOWLEDGE
    return TaskType.SIMPLE


_TASK_KEY: Dict[TaskType, str] = {
    TaskType.SIMPLE: "chat",
    TaskType.KNOWLEDGE: "memory",
    TaskType.MATH: "math",
    TaskType.CODE: "code",
    TaskType.REASONING: "planning",
    TaskType.ACTION: "action",
    TaskType.FULL: "planning",
}

_WAKE_MAP: Dict[TaskType, Set[str]] = {
    TaskType.SIMPLE: {"cerebellum", "thalamus"},
    TaskType.KNOWLEDGE: {"temporal", "thalamus", "hippocampus"},
    TaskType.MATH: {"parietal", "prefrontal", "thalamus", "cerebellum"},
    TaskType.CODE: {"prefrontal", "cerebellum", "temporal", "thalamus", "motor"},
    TaskType.REASONING: {"prefrontal", "parietal", "temporal", "thalamus", "cerebellum"},
    TaskType.ACTION: {"motor", "prefrontal", "cerebellum", "brainstem", "thalamus"},
    TaskType.FULL: {
        "prefrontal", "parietal", "temporal", "cerebellum",
        "hippocampus", "thalamus", "motor", "brainstem",
    },
}

_BUDGET_MAP: Dict[TaskType, float] = {
    TaskType.SIMPLE: 0.10,
    TaskType.KNOWLEDGE: 0.25,
    TaskType.MATH: 0.35,
    TaskType.CODE: 0.40,
    TaskType.REASONING: 0.55,
    TaskType.ACTION: 0.45,
    TaskType.FULL: 1.0,
}

_CTM_TICKS_MAP: Dict[TaskType, int] = {
    TaskType.SIMPLE: 1,
    TaskType.KNOWLEDGE: 2,
    TaskType.MATH: 3,
    TaskType.CODE: 3,
    TaskType.REASONING: 4,
    TaskType.ACTION: 2,
    TaskType.FULL: 4,
}


class ThalamusRouter(nn.Module):
    """丘脑路由器：文本/hidden/JEPA → RoutePlan（含真调度开关）。

    默认推理走 plan_from_jepa（JEPA 总控）；启发式仅作冷启动/对照。
    """

    def __init__(self, config: LunaConfig):
        super().__init__()
        self.config = config
        self.d_model = config.hidden_size
        n_types = len(TaskType)
        self.route_head = nn.Linear(self.d_model, n_types, bias=False)
        self._type_list = list(TaskType)

    def _plan(self, t: TaskType, reason: str, *,
              compute_scale: Optional[float] = None,
              wake_override: Optional[Set[str]] = None) -> RoutePlan:
        wake = set(wake_override) if wake_override is not None else set(_WAKE_MAP[t])
        wake.add("thalamus")
        subs = [s.value for s in subregions_for_task(_TASK_KEY[t])]
        scale = 1.0 if compute_scale is None else float(compute_scale)
        # JEPA compute_scale 调制预算与 ticks
        budget = min(1.0, max(0.05, _BUDGET_MAP[t] * (0.5 + scale)))
        ticks = max(1, int(round(_CTM_TICKS_MAP[t] * (0.5 + scale))))
        ticks = min(ticks, self.config.ctm_max_ticks)
        return RoutePlan(
            task_type=t,
            wake=wake,
            sub_regions=subs,
            expert_budget=budget,
            ctm_ticks=ticks,
            enable_layer_skip=(t in (TaskType.SIMPLE, TaskType.KNOWLEDGE)) or (scale < 0.35),
            enable_rag=(t in (TaskType.KNOWLEDGE, TaskType.CODE, TaskType.FULL)),
            enable_parietal=(t in (TaskType.MATH, TaskType.REASONING, TaskType.FULL, TaskType.CODE)),
            enable_hippo=(t in (TaskType.REASONING, TaskType.ACTION, TaskType.FULL, TaskType.KNOWLEDGE)),
            enable_cerebellum=True,
            enable_action=(t in (TaskType.ACTION, TaskType.CODE, TaskType.FULL)),
            cache_lookup=(t == TaskType.SIMPLE) or (scale < 0.25),
            reason=reason,
        )

    def plan_from_text(self, text: str) -> RoutePlan:
        t = classify_task(text)
        return self._plan(t, f"heuristic:{t.value}")

    def plan_from_hidden(self, prompt_hidden: torch.Tensor) -> RoutePlan:
        pooled = prompt_hidden.mean(dim=1)
        logits = self.route_head(pooled)
        idx = int(logits[0].argmax().item())
        t = self._type_list[idx]
        return self._plan(t, f"learned:{t.value}")

    def plan_from_jepa(self, signals) -> RoutePlan:
        """JEPA 总控路径：ControlSignals → RoutePlan。

        task 由 task_logits argmax；wake 由 wake_logits > 0 的集合（至少含 thalamus）；
        compute_scale 调制 ticks/budget。
        """
        from modeling_jepa_control import TASK_NAMES, WAKE_NAMES
        task_idx = int(signals.task_logits[0].argmax().item())
        task_idx = max(0, min(task_idx, len(self._type_list) - 1))
        # TASK_NAMES 与 TaskType 值对齐
        name = TASK_NAMES[task_idx] if task_idx < len(TASK_NAMES) else "simple"
        t = next((x for x in self._type_list if x.value == name), TaskType.SIMPLE)
        # wake：sigmoid > 0.5 的脑区；保证至少 thalamus + 默认 map 交集
        wake_probs = torch.sigmoid(signals.wake_logits[0])
        wake = {WAKE_NAMES[i] for i, p in enumerate(wake_probs.tolist()) if p > 0.5}
        if not wake:
            wake = set(_WAKE_MAP[t])
        wake |= {"thalamus"}
        # 与任务默认 wake 取并集，避免 JEPA 未训时全灭
        wake |= set(_WAKE_MAP[t])
        return self._plan(
            t,
            f"jepa:{t.value}:unc={signals.uncertainty:.2f}:scale={signals.compute_scale:.2f}",
            compute_scale=signals.compute_scale,
            wake_override=wake,
        )

    def expert_top_k(self, config: LunaConfig, plan: RoutePlan) -> int:
        base = config.num_expert_activated
        k = max(1, int(round(base * plan.expert_budget * 2)))
        return min(k, config.num_routed_experts)
