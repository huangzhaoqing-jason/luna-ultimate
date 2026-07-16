"""脑干动作门：动作级安全分类 + 拦截。

复用 safety/locks.py 的文本门，并加动作级分类：
  - 只读（read）：放行
  - 写入（write）：二次审（operator token / 审计）
  - 高危（danger）：阻断（rm -rf / force push / 外网恶意 / 删仓库 / drop database）
所有动作审计入 safety/audit.py。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from modeling_motor import ActionSpec


@dataclass
class ActionDecision:
    allowed: bool
    permission: str
    reason: str
    audit_id: Optional[str] = None


# 高危动作模式（危险系统命令 / 删仓库 / force push / 外网恶意）
_DANGER_PATTERNS = [
    re.compile(r"\brm\s+-rf\b", re.I),
    re.compile(r"\b(rm|del)\b .{0,20}\b(/|C:\\|root|home)\b", re.I),
    re.compile(r"\bgit\s+push\b", re.I),                 # 默认拦所有 git push（沙箱）
    re.compile(r"\bpush\b .{0,20}(--force|-f\b)", re.I), # force push 任意拼接
    re.compile(r"\b(drop|delete)\b .{0,20}\b(database|table|cluster)\b", re.I),
    re.compile(r"\b(sudo|chmod|chown)\b .{0,20}777", re.I),
    re.compile(r"\b(curl|wget)\b .{0,40}\|\s*(sh|bash)", re.I),
    re.compile(r"\b(format|mkfs)\b", re.I),
    re.compile(r"\b(nuke|wipe|destroy)\b .{0,20}\b(repo|repository|warehouse)\b", re.I),
    re.compile(r"\b:()\{\s*:|:&\}\;:", re.I),  # fork bomb
    re.compile(r"\b(eval|exec)\b .{0,20}\b(http|curl|wget)", re.I),
]

# 写入动作模式
_WRITE_PATTERNS = [
    re.compile(r"\b(git\s+(add|commit|push|merge|rebase))\b", re.I),
    re.compile(r"\b(write|edit|patch|apply|install|pip|npm|cargo)\b", re.I),
    re.compile(r"\b(file_write|file_edit)\b", re.I),
    re.compile(r"\b(mkdir|mv|cp)\b", re.I),
]


def _flatten_args(args: dict) -> str:
    """把 args 展平成可被 regex 扫描的字符串（处理 list/嵌套）。"""
    parts: List[str] = []
    for v in args.values():
        if isinstance(v, (list, tuple)):
            parts.extend(str(x) for x in v)
        elif isinstance(v, dict):
            for vv in v.values():
                parts.append(str(vv))
        else:
            parts.append(str(v))
    return " ".join(parts)


def classify_action(spec: ActionSpec) -> str:
    """返回 permission: read / write / danger。"""
    text = f"{spec.kind} {spec.name} " + _flatten_args(spec.args)
    for p in _DANGER_PATTERNS:
        if p.search(text):
            return "danger"
    if spec.permission == "danger":
        return "danger"
    for p in _WRITE_PATTERNS:
        if p.search(text):
            return "write"
    if spec.permission == "write":
        return "write"
    return "read"


class ActionGate:
    """脑干动作门。包装 SafetyLock + 动作级分类。"""

    def __init__(self, safety_lock, require_operator_for_write: bool = False):
        self.lock = safety_lock
        self.require_operator_for_write = require_operator_for_write

    def gate(
        self,
        spec: ActionSpec,
        operator_token: Optional[str] = None,
        context: Optional[dict] = None,
    ) -> ActionDecision:
        perm = classify_action(spec)
        from action.action_gate import _flatten_args  # noqa
        text = f"{spec.kind}:{spec.name} " + _flatten_args(spec.args)[:300]
        # 1. 文本门（charter/values/ctm）
        dec = self.lock.gate(text, operator_token=operator_token, context=context)
        if not dec.allowed:
            return ActionDecision(False, perm, f"refused by safety: {dec.reason}", dec.audit_id)
        # 2. 动作级门
        if perm == "danger":
            entry = self.lock._audit(
                actor=(dec.operator.operator_id if dec.operator else "anonymous"),
                action=text, decision="refuse",
                reason="Refused by brainstem action gate: dangerous action pattern.",
                context={"action_kind": spec.kind, "permission": perm},
            )
            return ActionDecision(False, perm,
                                  "Refused: dangerous action (rm -rf / force push / wipe / fork bomb / etc.).",
                                  entry.entry_id)
        if perm == "write" and self.require_operator_for_write and dec.operator is None:
            entry = self.lock._audit(
                actor="anonymous", action=text, decision="refuse",
                reason="Refused: write action requires operator authorization.",
                context={"action_kind": spec.kind, "permission": perm},
            )
            return ActionDecision(False, perm,
                                  "Refused: write action requires operator token.",
                                  entry.entry_id)
        return ActionDecision(True, perm, "Allowed.", dec.audit_id)
