"""Patch proposer and applicator for automated programming."""

from __future__ import annotations

import difflib
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class Patch:
    path: str
    before: str
    after: str
    rationale: str = ""

    def diff(self) -> str:
        return "\n".join(
            difflib.unified_diff(
                self.before.splitlines(keepends=True),
                self.after.splitlines(keepends=True),
                fromfile=f"a/{self.path}",
                tofile=f"b/{self.path}",
            )
        )


@dataclass
class PatchProposer:
    """Generates candidate patches from a mutator callable.

    The mutator receives (path, source) and returns either a new source
    string or None to skip. Proposers are intentionally simple and bounded:
    real deployments can swap in an LLM proposer; the safety gate still
    applies.
    """

    root: Path
    mutator: Optional[callable] = None
    target_globs: Tuple[str, ...] = ("evolve/*.py", "cost_model.py", "scripts/*.py")

    def _iter_targets(self) -> List[Path]:
        targets: List[Path] = []
        for g in self.target_globs:
            targets.extend(sorted(self.root.glob(g)))
        return targets

    def propose(self, max_proposals: int = 5) -> List[Patch]:
        mut = self.mutator or _default_mutator
        proposals: List[Patch] = []
        for p in self._iter_targets():
            if len(proposals) >= max_proposals:
                break
            try:
                src = p.read_text(encoding="utf-8")
            except Exception:
                continue
            new_src = mut(str(p.relative_to(self.root)), src)
            if new_src is None or new_src == src:
                continue
            proposals.append(Patch(
                path=str(p.relative_to(self.root)),
                before=src,
                after=new_src,
                rationale="auto-generated candidate patch",
            ))
        return proposals


def _default_mutator(path: str, source: str) -> Optional[str]:
    """Tiny built-in mutator: insert a no-op audit comment to prove the loop.

    This is deliberately trivial — it exercises the full pipeline
    (propose → gate → sandbox test → accept/revert) without requiring an LLM.
    """
    if not path.endswith(".py"):
        return None
    if "# luna-evolve: audited" in source:
        return None
    lines = source.splitlines(keepends=True)
    # Insert after the module docstring (or first line) to minimize breakage
    insert_at = 0
    if lines and lines[0].lstrip().startswith('"""'):
        for i in range(1, len(lines)):
            if '"""' in lines[i]:
                insert_at = i + 1
                break
    new_lines = lines[:insert_at] + ["# luna-evolve: audited (no-op marker)\n"] + lines[insert_at:]
    return "".join(new_lines)


def apply_patch(root: Path, patch: Patch) -> None:
    target = root / patch.path
    target.parent.mkdir(parents=True, exist_ok=True)
    backup = target.with_suffix(target.suffix + ".luna_bak")
    if target.exists():
        shutil.copy2(target, backup)
    target.write_text(patch.after, encoding="utf-8")


def accept_patch(root: Path, patch: Patch) -> None:
    """Call after a patch is accepted — removes the stale backup file."""
    target = root / patch.path
    backup = target.with_suffix(target.suffix + ".luna_bak")
    if backup.exists():
        backup.unlink()


def revert_patch(root: Path, patch: Patch) -> None:
    target = root / patch.path
    backup = target.with_suffix(target.suffix + ".luna_bak")
    if backup.exists():
        shutil.copy2(backup, target)
        backup.unlink()
    elif patch.before is not None:
        target.write_text(patch.before, encoding="utf-8")
