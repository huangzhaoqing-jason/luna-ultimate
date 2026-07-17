"""Compose atlas micro-macro + AIXI schedule + SiFu speech into one white box."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from brain.atlas.analyzer import MicroMacroReport
from brain.aixi.scheduler import ScheduleTrace
from brain.speech.sifu import WhiteBoxTrace


@dataclass
class WhiteBoxReasoning:
    """End-to-end inspectable cognition — AIXI-directed, never a black box."""

    atlas: Optional[MicroMacroReport] = None
    schedule: Optional[ScheduleTrace] = None
    speech: Optional[WhiteBoxTrace] = None
    aixi_action_index: Optional[int] = None
    aixi_expected_return: Optional[float] = None
    paradigm: str = "AIXI-schedule × BriLLM/SiFu × Brainnetome246"
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
            "schedule": self.schedule.to_dict() if self.schedule else None,
            "atlas": self.atlas.to_dict() if self.atlas else None,
            "speech": self.speech.to_dict() if self.speech else None,
        }

    def explain(self) -> str:
        parts = [f"=== WhiteBox Reasoning ({self.paradigm}) ==="]
        parts.append("Refs: " + "; ".join(self.papers))
        if self.aixi_action_index is not None:
            parts.append(
                f"AIXI action={self.aixi_action_index} "
                f"E[return]={self.aixi_expected_return}"
            )
        if self.schedule:
            parts.append(self.schedule.explain())
        if self.atlas:
            parts.append(self.atlas.explain())
        if self.speech:
            parts.append(self.speech.explain())
        return "\n".join(parts)
