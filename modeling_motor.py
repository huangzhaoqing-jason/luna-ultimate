"""运动动作中枢：M1 + 前运动皮层 + SMA。

- M1ActionHead：把高层规划转成结构化 ActionSpec（kind/args/timeout/retries/permission）。
- PremotorPlanner：把多动作编排成 DAG（serial/parallel/if-else/retry）。
- SMACache：高频动作模板缓存（命中即跳 M1 生成，最省算力）。

执行在 action/executor.py；门控在 action/action_gate.py（脑干动作门）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import LunaConfig
from modeling_vla import ActionHead, ActionTokenizer, DEFAULT_ACTION_VOCAB


# 动作类型
ACTION_KINDS = ("terminal", "git", "api", "file", "mcp")
PERMISSION_LEVELS = ("read", "write", "danger")


@dataclass
class ActionSpec:
    kind: str               # terminal / git / api / file / mcp
    name: str               # 动作名（如 git_commit / file_write / terminal_run）
    args: Dict[str, Any] = field(default_factory=dict)
    timeout: float = 30.0
    retries: int = 0
    permission: str = "read"   # read / write / danger
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionDAG:
    """前运动皮层编排的动作 DAG（最小：serial/parallel/retry）。"""
    nodes: List[ActionSpec] = field(default_factory=list)
    edges: List[Dict[str, Any]] = field(default_factory=list)  # {from, to, cond}
    retry_on_fail: int = 0
    parallel: bool = False


class M1ActionHead(nn.Module):
    """初级运动皮层：hidden → 结构化 ActionSpec（多分类 + 参数头）。"""

    def __init__(self, config: LunaConfig):
        super().__init__()
        self.d_model = config.hidden_size
        self.tokenizer = ActionTokenizer()
        # 动作名分类
        self.name_head = nn.Linear(self.d_model, len(self.tokenizer), bias=False)
        # 动作类型分类
        self.kind_head = nn.Linear(self.d_model, len(ACTION_KINDS), bias=False)
        # 权限分类
        self.perm_head = nn.Linear(self.d_model, len(PERMISSION_LEVELS), bias=False)
        # 超时/重试回归
        self.timeout_head = nn.Linear(self.d_model, 1, bias=False)
        self.retries_head = nn.Linear(self.d_model, 1, bias=False)
        # args 占位：用一个线性层产出「args 哈希键」（真实 args 由 serve/模板填充）
        self.args_head = nn.Linear(self.d_model, 8, bias=False)

    def forward(self, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        pooled = hidden.mean(dim=1)
        return {
            "name_logits": self.name_head(pooled),
            "kind_logits": self.kind_head(pooled),
            "perm_logits": self.perm_head(pooled),
            "timeout": self.timeout_head(pooled).squeeze(-1),
            "retries": self.retries_head(pooled).squeeze(-1),
            "args_hash": self.args_head(pooled),
        }


# SMA 模板：高频工作流（命中即跳 M1 生成）
SMA_TEMPLATES: Dict[str, ActionDAG] = {
    "git_commit_local": ActionDAG(
        nodes=[
            ActionSpec("git", "git_status", {"args": ["status"]}, permission="read"),
            ActionSpec("git", "git_add", {"args": ["add", "-A"]}, permission="write"),
            ActionSpec("git", "git_commit", {"args": ["commit", "-m", "luna action"]}, permission="write"),
        ],
        edges=[{"from": 0, "to": 1, "cond": None}, {"from": 1, "to": 2, "cond": None}],
        retry_on_fail=1,
    ),
    "lora_finetune": ActionDAG(
        nodes=[
            ActionSpec("terminal", "lora_run",
                       {"args": ["python3", "scripts/lora_finetune.py", "--preset", "tiny"]},
                       permission="write", timeout=120),
        ],
        retry_on_fail=1,
    ),
    "mcp_selfcheck": ActionDAG(
        nodes=[
            ActionSpec("mcp", "mcp_health", {}, permission="read"),
            ActionSpec("mcp", "mcp_audit", {}, permission="read"),
        ],
        edges=[{"from": 0, "to": 1, "cond": None}],
    ),
    "read_file": ActionDAG(
        nodes=[ActionSpec("file", "file_read", {}, permission="read")],
    ),
}


class SMACache:
    """辅助运动区：高频动作模板缓存。"""

    def __init__(self, templates: Optional[Dict[str, ActionDAG]] = None):
        self.templates = dict(templates or SMA_TEMPLATES)

    def lookup(self, intent: str) -> Optional[ActionDAG]:
        s = (intent or "").lower()
        for key, dag in self.templates.items():
            if key in s or any(w in s for w in key.split("_")):
                return dag
        return None

    def register(self, name: str, dag: ActionDAG) -> None:
        self.templates[name] = dag


class PremotorPlanner(nn.Module):
    """前运动皮层：把 M1 输出 + SMA 模板编排成 ActionDAG。"""

    def __init__(self, config: LunaConfig):
        super().__init__()
        self.d_model = config.hidden_size
        self.sma = SMACache()
        # 编排骨干：hidden → DAG 选择分数（占位，4 种骨架）
        self.skeleton_head = nn.Linear(self.d_model, 4, bias=False)

    def forward(
        self,
        hidden: torch.Tensor,
        m1_out: Dict[str, torch.Tensor],
        intent: str,
    ) -> ActionDAG:
        # SMA 命中优先（最省算力）
        cached = self.sma.lookup(intent)
        if cached is not None:
            return cached
        pooled = hidden.mean(dim=1)
        skel_idx = int(self.skeleton_head(pooled)[0].argmax().item())
        name_idx = int(m1_out["name_logits"][0].argmax().item())
        kind_idx = int(m1_out["kind_logits"][0].argmax().item())
        perm_idx = int(m1_out["perm_logits"][0].argmax().item())
        timeout = float(m1_out["timeout"][0].item())
        retries = int(max(0, m1_out["retries"][0].item()))
        spec = ActionSpec(
            kind=ACTION_KINDS[kind_idx],
            name=self.tokenizer.decode(name_idx),
            args={},
            timeout=max(5.0, min(timeout, 300.0)),
            retries=min(retries, 3),
            permission=PERMISSION_LEVELS[perm_idx],
            meta={"skeleton": skel_idx, "intent": intent[:120]},
        )
        return ActionDAG(nodes=[spec], retry_on_fail=min(retries, 2))


class MotorCortex(nn.Module):
    """运动动作中枢：M1 → premotor → SMA。"""

    def __init__(self, config: LunaConfig):
        super().__init__()
        self.m1 = M1ActionHead(config)
        self.premotor = PremotorPlanner(config)

    def plan_actions(self, hidden: torch.Tensor, intent: str) -> Dict[str, Any]:
        m1_out = self.m1(hidden)
        dag = self.premotor(hidden, m1_out, intent=intent)
        return {
            "m1": {
                "name_idx": int(m1_out["name_logits"][0].argmax().item()),
                "kind": ACTION_KINDS[int(m1_out["kind_logits"][0].argmax().item())],
                "permission": PERMISSION_LEVELS[int(m1_out["perm_logits"][0].argmax().item())],
                "timeout": float(m1_out["timeout"][0].item()),
                "retries": int(m1_out["retries"][0].item()),
            },
            "dag": dag,
            "sma_hit": dag is not None and any(n.name in {t.name for d in self.premotor.sma.templates.values() for t in d.nodes} for n in dag.nodes),
        }
