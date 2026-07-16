"""Sandboxed subprocess runner for testing generated code.

Hard rules:
  - no network (best-effort: subprocess + env scrubbing + no requests import
    in the harness; full network isolation requires OS-level sandboxing the
    operator must configure)
  - wall-clock timeout
  - separate process so a crash does not kill the evolve loop
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class SandboxResult:
    returncode: int
    stdout: str
    stderr: str
    elapsed_s: float
    timed_out: bool = False
    command: List[str] = field(default_factory=list)


def _scrub_env() -> dict:
    env = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/tmp"),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "LUNA_SANDBOX": "1",
    }
    # Strip proxy / token env vars to reduce exfil risk from generated code
    for k in list(os.environ.keys()):
        if any(s in k.upper() for s in ("TOKEN", "SECRET", "KEY", "CRED", "PROXY", "AUTH")):
            continue
    return env


def run_in_sandbox(
    command: List[str],
    cwd: Path,
    timeout_s: float = 30.0,
    network_disabled: bool = True,
) -> SandboxResult:
    """Run a command in a subprocess with a hard timeout and scrubbed env.

    Note: true network isolation requires OS-level sandboxing (e.g. firejail,
    namespace, seccomp). We disable common network libs via env and rely on
    the operator to enforce network policy. Generated code MUST NOT be given
    real network access.
    """
    env = _scrub_env()
    if network_disabled:
        # Hint to harness; not a hard guarantee
        env["LUNA_NETWORK"] = "disabled"

    t0 = time.time()
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        return SandboxResult(
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            elapsed_s=time.time() - t0,
            command=command,
        )
    except subprocess.TimeoutExpired as e:
        return SandboxResult(
            returncode=-1,
            stdout=(e.stdout or "") if isinstance(e.stdout, str) else "",
            stderr=(e.stderr or "") if isinstance(e.stderr, str) else "",
            elapsed_s=timeout_s,
            timed_out=True,
            command=command,
        )


def run_pytest_in_sandbox(
    cwd: Path,
    test_targets: List[str],
    timeout_s: float = 60.0,
) -> SandboxResult:
    """Run pytest on the given targets in a sandbox."""
    cmd = [sys.executable, "-m", "pytest", "-q", *test_targets]
    return run_in_sandbox(cmd, cwd=cwd, timeout_s=timeout_s)


def run_safety_tests_in_sandbox(cwd: Path, timeout_s: float = 30.0) -> SandboxResult:
    """Run the safety test suite in a sandbox subprocess."""
    cmd = [sys.executable, "-c",
           "from safety.tests import SafetyTestSuite; "
           "r = SafetyTestSuite().run(); "
           "import sys; sys.exit(0 if all(x.passed for x in r) else 1)"]
    return run_in_sandbox(cmd, cwd=cwd, timeout_s=timeout_s)
