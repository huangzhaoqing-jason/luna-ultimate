"""顶叶：空间 / 数值 / 结构化推理（神经符号骨架）。

轻量实现：神经直觉头 + 符号校验占位（regex/AST）。
预留 Lean/Isabelle 接口，不在本地冒烟中依赖外部证明器。
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import LunaConfig


_NUM_EXPR = re.compile(
    r"^\s*[-+]?(?:\d+(?:\.\d*)?|\.\d+)\s*(?:[-+*/^%]\s*[-+]?(?:\d+(?:\.\d*)?|\.\d+)\s*)+$"
)


@dataclass
class ParietalResult:
    ok: bool
    kind: str
    detail: str
    value: Optional[float] = None


class ParietalReasoner(nn.Module):
    """神经直觉 + 轻量符号验证。"""

    def __init__(self, config: LunaConfig):
        super().__init__()
        self.d_model = config.hidden_size
        self.intuit = nn.Sequential(
            nn.Linear(self.d_model, self.d_model, bias=False),
            nn.GELU(),
            nn.Linear(self.d_model, 1, bias=False),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden [B, L, d] → 数值直觉分数 [B, 1]。"""
        pooled = hidden.mean(dim=1)
        return self.intuit(pooled)

    def verify_expression(self, text: str) -> ParietalResult:
        """对简单算术表达式做符号校验（占位）。"""
        t = (text or "").strip()
        if not t:
            return ParietalResult(False, "empty", "empty expression")
        # AST 安全求值：仅允许字面量与二元运算
        try:
            tree = ast.parse(t, mode="eval")
            if not self._is_safe_ast(tree):
                return ParietalResult(False, "unsafe_ast", "expression not a pure numeric AST")
            val = float(eval(compile(tree, "<parietal>", "eval"), {"__builtins__": {}}, {}))
            return ParietalResult(True, "numeric", "ok", value=val)
        except Exception as e:
            if _NUM_EXPR.match(t):
                return ParietalResult(False, "parse_fail", f"regex-ok but eval failed: {e}")
            return ParietalResult(False, "not_numeric", str(e))

    @staticmethod
    def _is_safe_ast(node: ast.AST) -> bool:
        allowed = (
            ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
            ast.UAdd, ast.USub, ast.FloorDiv,
        )
        for n in ast.walk(node):
            if not isinstance(n, allowed):
                return False
        return True

    def reason(
        self, hidden: Optional[torch.Tensor] = None, text: Optional[str] = None
    ) -> Dict[str, object]:
        out: Dict[str, object] = {"module": "parietal"}
        if hidden is not None:
            out["intuition"] = float(self.forward(hidden).mean().item())
        if text is not None:
            vr = self.verify_expression(text)
            out["verify"] = {
                "ok": vr.ok, "kind": vr.kind, "detail": vr.detail, "value": vr.value
            }
        return out
