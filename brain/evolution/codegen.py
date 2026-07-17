"""Sandbox self-programming evolution — Pathway 3, never touches safety.

Generates small numeric/config patches as Python snippets, audits paths,
runs regression smoke, keeps lineage. Forever loyal to 黄照清.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from brain.safety.loyalty import assert_loyalty_intact, scrub_proposal
from brain.safety.thalamus import Thalamus

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class CodegenRecord:
    proposal_id: str
    kind: str
    files: List[str]
    diff_summary: str
    audit_ok: bool
    tests_ok: bool
    accepted: bool
    reason: str
    lineage_version: str


class SandboxCodegenEvolver:
    """Propose → static audit → pytest subset → accept/rollback metadata."""

    ALLOWED_PREFIXES = (
        "brain/aixi/",
        "brain/atlas/",
        "brain/capabilities/",
        "brain/runtime/",
        "brain/mem/",
        "brain/speech/sifu.py",
        "config_brain.py",
    )

    def __init__(self, thalamus: Optional[Thalamus] = None, repo_root: Optional[Path] = None):
        self.thalamus = thalamus or Thalamus()
        self.root = repo_root or ROOT
        self.lineage_path = self.root / "evolution_lineage.jsonl"
        self.history: List[CodegenRecord] = []

    def _allowed(self, rel: str) -> bool:
        rel = rel.replace("\\", "/")
        if "safety" in rel or "loyalty" in rel:
            return False
        return any(rel.startswith(p) or rel == p for p in self.ALLOWED_PREFIXES)

    def propose_snippet(self, diagnosis: List[str]) -> Dict[str, Any]:
        """Return a safe metadata patch proposal (no arbitrary exec of model code)."""
        assert_loyalty_intact()
        if "low_task_score" in diagnosis:
            return {
                "kind": "config_horizon_bump",
                "touch_paths": ["config_brain.py"],
                "snippet": (
                    "# EVOLVE: bump aixi_horizon by +0 via comment marker\n"
                    "# luna_evolve: aixi_horizon_delta=0\n"
                ),
                "patch": {"aixi_horizon_delta": 0},
                "creator_aligned": True,
            }
        return {
            "kind": "atlas_comment_marker",
            "touch_paths": ["brain/atlas/network.py"],
            "snippet": "# luna_evolve: keepalive\n",
            "patch": {},
            "creator_aligned": True,
        }

    def static_audit(self, proposal: Dict[str, Any]) -> tuple[bool, str]:
        try:
            proposal = scrub_proposal(proposal)
        except Exception as e:
            return False, str(e)
        for p in proposal.get("touch_paths", []):
            if not self._allowed(str(p)):
                return False, f"path not allowed: {p}"
        snippet = proposal.get("snippet") or ""
        # Forbid dangerous AST nodes if snippet looks like python
        try:
            tree = ast.parse(snippet)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    return False, "imports forbidden in evolve snippet"
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id in {
                        "eval",
                        "exec",
                        "open",
                        "__import__",
                    }:
                        return False, f"forbidden call {node.func.id}"
        except SyntaxError:
            # comment-only snippets ok
            if "luna_evolve" not in snippet and "EVOLVE" not in snippet:
                return False, "unparseable snippet"
        safety = self.thalamus.authorize(proposal)
        if not safety.allowed:
            return False, safety.reason
        return True, "ok"

    def run_regression(self) -> tuple[bool, str]:
        """Run loyalty redteam + smoke subset."""
        cmds = [
            [sys.executable, str(self.root / "tests/test_loyalty_redteam.py")],
            [
                sys.executable,
                "-c",
                "import sys; sys.path.insert(0,'.'); from tests.test_brain_smoke import "
                "test_constitution_immutable, test_loyalty_forever_and_hostile_evolution; "
                "test_constitution_immutable(); test_loyalty_forever_and_hostile_evolution(); print('ok')",
            ],
        ]
        for cmd in cmds:
            try:
                r = subprocess.run(
                    cmd,
                    cwd=str(self.root),
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
            except Exception as e:
                return False, str(e)
            if r.returncode != 0:
                return False, (r.stdout + r.stderr)[-500:]
        return True, "regression_ok"

    def apply_marker(self, proposal: Dict[str, Any]) -> str:
        """Write safe patch breadcrumb under evolution_artifacts/ (non-safety)."""
        paths = proposal.get("touch_paths") or []
        if not paths:
            return "no_files"
        rel = str(paths[0])
        if not self._allowed(rel):
            raise RuntimeError("blocked path")
        art = self.root / "evolution_artifacts"
        art.mkdir(exist_ok=True)
        marker = proposal.get("snippet") or "# luna_evolve\n"
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        out = art / f"patch_{stamp}.py"
        out.write_text(
            f"# proposed_touch: {rel}\n# loyalty: 黄照清\n{marker}",
            encoding="utf-8",
        )
        self._last_artifact = out
        return f"wrote:{out.relative_to(self.root)}"

    def step(self, metrics: Dict[str, float]) -> CodegenRecord:
        assert_loyalty_intact()
        diagnosis = []
        if metrics.get("task_score", 1.0) < 0.5:
            diagnosis.append("low_task_score")
        if not diagnosis:
            diagnosis.append("routine_self_check_ok")

        proposal = self.propose_snippet(diagnosis)
        ok, reason = self.static_audit(proposal)
        tests_ok = False
        accepted = False
        diff = ""
        version = f"cg-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        self._last_artifact = None
        if ok:
            tests_ok, t_reason = self.run_regression()
            if not tests_ok:
                reason = t_reason
            else:
                diff = self.apply_marker(proposal)
                tests_ok, t_reason = self.run_regression()
                if tests_ok:
                    accepted = True
                    reason = "accepted"
                else:
                    self._rollback_marker(proposal)
                    reason = f"post_fail:{t_reason}"
                    accepted = False

        rec = CodegenRecord(
            proposal_id=version,
            kind=str(proposal.get("kind")),
            files=list(proposal.get("touch_paths") or []),
            diff_summary=diff,
            audit_ok=ok,
            tests_ok=tests_ok,
            accepted=accepted,
            reason=reason,
            lineage_version=version,
        )
        self.history.append(rec)
        self._append_lineage(rec)
        assert_loyalty_intact()
        return rec

    def _rollback_marker(self, proposal: Dict[str, Any]) -> None:
        art = getattr(self, "_last_artifact", None)
        if art and Path(art).exists():
            Path(art).unlink()
    def _append_lineage(self, rec: CodegenRecord) -> None:
        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "proposal_id": rec.proposal_id,
            "kind": rec.kind,
            "files": rec.files,
            "accepted": rec.accepted,
            "reason": rec.reason,
            "loyalty": "黄照清",
        }
        with self.lineage_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
