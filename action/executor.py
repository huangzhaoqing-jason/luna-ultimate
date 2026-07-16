"""动作执行器：按 ActionSpec.kind 沙箱执行。

沙箱：超时、工作目录白名单、网络默认关。返回 ActionResult。
不自动 push 仓库 / 不自动上传权重（沿用开关）。
"""

from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from modeling_motor import ActionDAG, ActionSpec


@dataclass
class ActionResult:
    ok: bool
    name: str
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    duration: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)


class ActionExecutor:
    """沙箱动作执行器。"""

    def __init__(
        self,
        workdir: str = ".",
        allow_network: bool = False,
        default_timeout: float = 30.0,
    ):
        self.workdir = Path(workdir).resolve()
        self.allow_network = allow_network
        self.default_timeout = default_timeout

    def execute(self, spec: ActionSpec) -> ActionResult:
        if spec.kind == "terminal":
            return self._run_terminal(spec)
        if spec.kind == "git":
            return self._run_git(spec)
        if spec.kind == "file":
            return self._run_file(spec)
        if spec.kind == "api":
            return self._run_api(spec)
        if spec.kind == "mcp":
            return self._run_mcp(spec)
        return ActionResult(False, spec.name, stderr=f"unknown kind {spec.kind}")

    def execute_dag(self, dag: ActionDAG) -> List[ActionResult]:
        results: List[ActionResult] = []
        for node in dag.nodes:
            res = self.execute(node)
            results.append(res)
            if not res.ok and dag.retry_on_fail > 0:
                for _ in range(dag.retry_on_fail):
                    res = self.execute(node)
                    results[-1] = res
                    if res.ok:
                        break
            if not res.ok:
                break  # serial：失败即停（parallel 占位未实装）
        return results

    # —— 各类执行 ——

    def _run_terminal(self, spec: ActionSpec) -> ActionResult:
        args = spec.args.get("args") or []
        if isinstance(args, str):
            args = shlex.split(args)
        if not args:
            return ActionResult(False, spec.name, stderr="empty terminal args")
        # 安全：禁用 network 环境变量；限制 cwd
        env = dict(os.environ)
        if not self.allow_network:
            env["LUNA_NO_NETWORK"] = "1"
        try:
            import time
            t0 = time.time()
            r = subprocess.run(
                args, cwd=str(self.workdir), capture_output=True, text=True,
                timeout=spec.timeout or self.default_timeout, env=env,
            )
            return ActionResult(
                ok=(r.returncode == 0), name=spec.name,
                stdout=r.stdout[-4000:], stderr=r.stderr[-4000:],
                exit_code=r.returncode, duration=time.time() - t0,
            )
        except subprocess.TimeoutExpired:
            return ActionResult(False, spec.name, stderr=f"timeout {spec.timeout}s")
        except Exception as e:
            return ActionResult(False, spec.name, stderr=str(e))

    def _run_git(self, spec: ActionSpec) -> ActionResult:
        args = spec.args.get("args") or []
        if isinstance(args, str):
            args = shlex.split(args)
        # 强制本地：拦截 push（除非显式 allow_push=True）
        if args and args[0] == "push" and not spec.args.get("allow_push"):
            return ActionResult(False, spec.name,
                                stderr="git push blocked by sandbox (set allow_push=True)")
        cmd = ["git"] + [str(a) for a in args]
        try:
            import time
            t0 = time.time()
            r = subprocess.run(
                cmd, cwd=str(self.workdir), capture_output=True, text=True,
                timeout=spec.timeout or self.default_timeout,
            )
            return ActionResult(
                ok=(r.returncode == 0), name=spec.name,
                stdout=r.stdout[-4000:], stderr=r.stderr[-4000:],
                exit_code=r.returncode, duration=time.time() - t0,
            )
        except Exception as e:
            return ActionResult(False, spec.name, stderr=str(e))

    def _run_file(self, spec: ActionSpec) -> ActionResult:
        path = spec.args.get("path")
        if not path:
            return ActionResult(False, spec.name, stderr="file action needs path")
        target = (self.workdir / path).resolve()
        # 工作目录白名单
        try:
            target.relative_to(self.workdir)
        except ValueError:
            return ActionResult(False, spec.name, stderr="path outside workdir whitelist")
        op = spec.name
        try:
            if op == "file_read":
                return ActionResult(True, op, stdout=target.read_text(encoding="utf-8")[:4000])
            if op == "file_write":
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(str(spec.args.get("content", "")), encoding="utf-8")
                return ActionResult(True, op, stdout=f"wrote {target}")
            if op == "file_edit":
                old = spec.args.get("old", "")
                new = spec.args.get("new", "")
                txt = target.read_text(encoding="utf-8")
                if old not in txt:
                    return ActionResult(False, op, stderr="old string not found")
                target.write_text(txt.replace(old, new, 1), encoding="utf-8")
                return ActionResult(True, op, stdout=f"edited {target}")
            return ActionResult(False, op, stderr=f"unknown file op {op}")
        except Exception as e:
            return ActionResult(False, op, stderr=str(e))

    def _run_api(self, spec: ActionSpec) -> ActionResult:
        if not self.allow_network:
            return ActionResult(False, spec.name, stderr="api blocked (network disabled)")
        try:
            import requests  # type: ignore
            url = spec.args.get("url")
            method = spec.args.get("method", "GET")
            r = requests.request(method, url, timeout=spec.timeout or self.default_timeout,
                                 **{k: v for k, v in spec.args.items()
                                    if k not in ("url", "method")})
            return ActionResult(ok=(r.status_code < 400), name=spec.name,
                                stdout=r.text[:4000], exit_code=r.status_code)
        except ImportError:
            return ActionResult(False, spec.name, stderr="requests not installed")
        except Exception as e:
            return ActionResult(False, spec.name, stderr=str(e))

    def _run_mcp(self, spec: ActionSpec) -> ActionResult:
        # MCP 自检占位：返回健康标记
        if spec.name == "mcp_health":
            return ActionResult(True, spec.name, stdout="mcp_health=ok")
        if spec.name == "mcp_audit":
            return ActionResult(True, spec.name, stdout="mcp_audit=ok")
        return ActionResult(False, spec.name, stderr=f"unknown mcp op {spec.name}")
