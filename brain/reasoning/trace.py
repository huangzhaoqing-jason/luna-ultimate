"""Compose atlas micro-macro + AIXI ledger + SiFu speech into one white box."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from brain.atlas.analyzer import MicroMacroReport
from brain.aixi.scheduler import ScheduleTrace
from brain.aixi.orchestrator import AIXILedger
from brain.speech.sifu import WhiteBoxTrace


@dataclass
class WhiteBoxReasoning:
    """End-to-end inspectable cognition — AIXI-directed, never a black box."""

    atlas: Optional[MicroMacroReport] = None
    schedule: Optional[ScheduleTrace] = None
    speech: Optional[WhiteBoxTrace] = None
    ledger: Optional[AIXILedger] = None
    all_areas: Optional[List[Dict[str, Any]]] = None
    aixi_action_index: Optional[int] = None
    aixi_expected_return: Optional[float] = None
    paradigm: str = "AIXIOrchestrator × BriLLM/SiFu × Brainnetome246 (自研)"
    papers: List[str] = field(
        default_factory=lambda: [
            "arXiv:2606.12683 From AGI to ASI",
            "Hutter AIXI / Universal AI",
            "arXiv:2503.11299 BriLLM (SiFu structure imitation, self-implemented)",
        ]
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "paradigm": self.paradigm,
            "papers": self.papers,
            "aixi_action_index": self.aixi_action_index,
            "aixi_expected_return": self.aixi_expected_return,
            "ledger": self.ledger.to_dict() if self.ledger else None,
            "schedule": self.schedule.to_dict() if self.schedule else None,
            "atlas": self.atlas.to_dict() if self.atlas else None,
            "speech": self.speech.to_dict() if self.speech else None,
            "all_areas_count": len(self.all_areas) if self.all_areas else 0,
        }

    def explain(self, include_all_areas: bool = False) -> str:
        parts = [f"=== WhiteBox Reasoning ({self.paradigm}) ==="]
        parts.append("Refs: " + "; ".join(self.papers))
        if self.aixi_action_index is not None:
            parts.append(
                f"AIXI action={self.aixi_action_index} "
                f"E[return]={self.aixi_expected_return}"
            )
        if self.ledger:
            parts.append(self.ledger.explain())
        if self.schedule:
            parts.append(self.schedule.explain())
        if self.atlas:
            parts.append(self.atlas.explain())
        if self.speech:
            parts.append(self.speech.explain())
        if include_all_areas and self.all_areas:
            parts.append(f"[Full246] dumping {len(self.all_areas)} area cards:")
            for row in self.all_areas:
                parts.append(
                    f"  #{row['area_id']:03d} act={row['activation']:.4f} "
                    f"spike={row['spike_rate']} [{row['macro']}] "
                    f"sched={row['scheduled']}\n"
                    f"    micro: {row['micro']}"
                )
        elif self.all_areas:
            parts.append(
                f"[Full246] {len(self.all_areas)} cards available via "
                f"to_dict()/all_areas (pass include_all_areas=True to print)"
            )
        return "\n".join(parts)
