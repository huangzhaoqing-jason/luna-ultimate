"""Tamper-evident hash-chained audit log."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AuditEntry:
    actor: str
    action: str
    decision: str
    reason: str
    context: Dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)
    prev_hash: str = ""
    entry_id: str = ""

    def compute_hash(self) -> str:
        payload = {
            "actor": self.actor,
            "action": self.action,
            "decision": self.decision,
            "reason": self.reason,
            "context": self.context,
            "ts": self.ts,
            "prev_hash": self.prev_hash,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()


class AuditLog:
    """Append-only, hash-chained audit log persisted to disk."""

    def __init__(self, path: str = ".luna/audit.log"):
        self.path = path
        self.entries: List[AuditEntry] = []
        self._last_hash = ""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):
            self._load()
        else:
            # Write a genesis entry so the chain has a known head
            genesis = AuditEntry(
                actor="system", action="audit_genesis",
                decision="allow", reason="audit log initialized",
            )
            genesis.prev_hash = "0" * 64
            genesis.entry_id = genesis.compute_hash()
            self._last_hash = genesis.entry_id
            self.entries.append(genesis)
            self._persist(genesis)

    def _load(self) -> None:
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                e = AuditEntry(**d)
                self.entries.append(e)
        if self.entries:
            self._last_hash = self.entries[-1].entry_id

    def _persist(self, entry: AuditEntry) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(entry)) + "\n")

    def append(self, entry: AuditEntry) -> AuditEntry:
        entry.prev_hash = self._last_hash
        entry.entry_id = entry.compute_hash()
        self._last_hash = entry.entry_id
        self.entries.append(entry)
        self._persist(entry)
        return entry

    def verify_chain(self) -> bool:
        prev = "0" * 64
        for e in self.entries:
            if e.prev_hash != prev:
                return False
            if e.compute_hash() != e.entry_id:
                return False
            prev = e.entry_id
        return True

    def recent(self, n: int = 20) -> List[AuditEntry]:
        return self.entries[-n:]
