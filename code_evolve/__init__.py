"""Luna Code Evolve: sandboxed, safety-gated automated programming.

The loop proposes small code patches, runs them in a sandboxed subprocess with
NO network access, runs the correctness tests + the safety test suite, and
only accepts patches that pass everything. Patches to safety/ are always
refused (immutable charter).
"""

from code_evolve.patcher import PatchProposer, apply_patch, revert_patch
from code_evolve.sandbox import run_in_sandbox, SandboxResult
from code_evolve.loop import CodeEvolveLoop, CodeEvolveResult

__all__ = [
    "PatchProposer",
    "apply_patch",
    "revert_patch",
    "run_in_sandbox",
    "SandboxResult",
    "CodeEvolveLoop",
    "CodeEvolveResult",
]
