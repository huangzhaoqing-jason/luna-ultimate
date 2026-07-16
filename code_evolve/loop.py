"""Code Evolve main loop: propose → safety-gate → sandbox test → accept/revert."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, List, Optional

from code_evolve.patcher import (
    PatchProposer,
    Patch,
    apply_patch,
    accept_patch,
    revert_patch,
)
from code_evolve.sandbox import (
    SandboxResult,
    run_in_sandbox,
    run_safety_tests_in_sandbox,
)
from safety.audit import AuditLog
from safety.charter import verify_charter
from safety.locks import SafetyLock
from safety.tests import SafetyTestSuite


@dataclass
class CodeEvolveResult:
    accepted: int = 0
    refused: int = 0
    failed: int = 0
    reverted: int = 0
    details: List[dict] = field(default_factory=list)


class CodeEvolveLoop:
    """Bounded, audited, safety-gated automated programming loop.

    Each iteration:
      1. verify charter integrity
      2. propose a patch
      3. SafetyLock.gate_code_patch (safety/ always refused)
      4. apply patch to a scratch copy
      5. run safety tests + correctness tests in a sandbox (no network)
      6. accept iff both pass; else revert
      7. audit every step
    """

    def __init__(
        self,
        root: Path,
        audit_log: Optional[AuditLog] = None,
        safety_lock: Optional[SafetyLock] = None,
        proposer: Optional[PatchProposer] = None,
        operator_token: Optional[str] = None,
        test_targets: Optional[List[str]] = None,
        sandbox_timeout_s: float = 60.0,
        safety_ctm: Optional[Any] = None,
    ):
        self.root = root
        self.audit = audit_log or AuditLog(str(root / ".luna" / "audit.log"))
        # Attach the CTM cognitive judge to the safety lock if provided
        if safety_lock is None and safety_ctm is not None:
            self.safety_lock = SafetyLock(
                self.audit, base_dir=str(root / ".luna"), safety_ctm=safety_ctm
            )
        else:
            self.safety_lock = safety_lock or SafetyLock(
                self.audit, base_dir=str(root / ".luna")
            )
        self.proposer = proposer or PatchProposer(root=root)
        self.operator_token = operator_token
        self.test_targets = test_targets or []
        self.sandbox_timeout_s = sandbox_timeout_s

    def _run_correctness(self) -> SandboxResult:
        if not self.test_targets:
            # Default: run a tiny self-check that imports the package
            cmd = [sys.executable, "-c",
                   "import config, cost_model, modeling_luna_ultimate; "
                   "from config import LunaConfig; "
                   "LunaConfig.from_preset('tiny'); print('ok')"]
            return run_in_sandbox(cmd, cwd=self.root, timeout_s=self.sandbox_timeout_s)
        cmd = [sys.executable, "-m", "pytest", "-q", *self.test_targets]
        return run_in_sandbox(cmd, cwd=self.root, timeout_s=self.sandbox_timeout_s)

    def _run_safety(self) -> SandboxResult:
        return run_safety_tests_in_sandbox(self.root, timeout_s=self.sandbox_timeout_s)

    def step(self, patch: Patch) -> dict:
        # Charter integrity
        verify_charter()

        # Safety gate
        decision = self.safety_lock.gate_code_patch(
            patch.path, patch.diff(), operator_token=self.operator_token
        )
        if not decision.allowed:
            return {"patch": patch.path, "outcome": "refused",
                    "reason": decision.reason, "audit_id": decision.audit_id}

        # Apply
        apply_patch(self.root, patch)
        try:
            safety_res = self._run_safety()
            corr_res = self._run_correctness()
            ok = (
                safety_res.returncode == 0
                and not safety_res.timed_out
                and corr_res.returncode == 0
                and not corr_res.timed_out
            )
            detail = {
                "patch": patch.path,
                "outcome": "accepted" if ok else "failed",
                "safety_rc": safety_res.returncode,
                "correctness_rc": corr_res.returncode,
                "elapsed_s": safety_res.elapsed_s + corr_res.elapsed_s,
                "stderr": (corr_res.stderr or safety_res.stderr)[:500],
            }
            if not ok:
                revert_patch(self.root, patch)
                detail["outcome"] = "reverted"
            else:
                accept_patch(self.root, patch)
            return detail
        except Exception as e:
            revert_patch(self.root, patch)
            return {"patch": patch.path, "outcome": "reverted",
                    "reason": f"exception: {e}"}

    def run(self, max_iterations: int = 5, max_proposals: int = 5) -> CodeEvolveResult:
        result = CodeEvolveResult()
        for _ in range(max_iterations):
            proposals = self.proposer.propose(max_proposals=max_proposals)
            if not proposals:
                break
            for patch in proposals:
                d = self.step(patch)
                result.details.append(d)
                outcome = d.get("outcome")
                if outcome == "accepted":
                    result.accepted += 1
                elif outcome == "refused":
                    result.refused += 1
                elif outcome == "reverted":
                    result.reverted += 1
                else:
                    result.failed += 1
        return result


def main():
    p = argparse.ArgumentParser(description="Luna Code Evolve (sandboxed, safety-gated)")
    p.add_argument("--root", type=str, default=".")
    p.add_argument("--max_iterations", type=int, default=3)
    p.add_argument("--max_proposals", type=int, default=3)
    p.add_argument("--operator_token", type=str, default=None,
                   help="Operator token (required for sensitive patches)")
    p.add_argument("--sandbox_timeout_s", type=float, default=60.0)
    p.add_argument("--test_targets", type=str, nargs="*", default=[])
    p.add_argument("--output", type=str, default="./code_evolve_run.json")
    p.add_argument("--no-ctm", action="store_true",
                   help="Disable the CTM cognitive judge (charter hard floor still applies)")
    args = p.parse_args()

    root = Path(args.root).resolve()
    verify_charter()
    SafetyTestSuite().run()

    safety_ctm = None
    if not args.no_ctm:
        from safety.cognition import default_safety_ctm
        safety_ctm = default_safety_ctm()

    loop = CodeEvolveLoop(
        root=root,
        operator_token=args.operator_token,
        test_targets=args.test_targets,
        sandbox_timeout_s=args.sandbox_timeout_s,
        safety_ctm=safety_ctm,
    )
    res = loop.run(max_iterations=args.max_iterations, max_proposals=args.max_proposals)
    payload = asdict(res)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
