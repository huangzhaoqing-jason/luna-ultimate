"""White-box micro→meso→macro activation report over 246 areas."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from brain.atlas.brainnetome246 import MACRO_SYSTEMS, NUM_AREAS
from brain.atlas.functions import FUNCTION_CARDS, card_for, macro_summary


@dataclass
class AreaExplain:
    area_id: int
    name: str
    macro: str
    activation: float
    micro: str
    meso: str
    macro_role: str
    aixi_hook: str
    scheduled: bool


@dataclass
class MicroMacroReport:
    """Fully inspectable 246-area parse — never a single opaque logit."""

    top_areas: List[AreaExplain] = field(default_factory=list)
    macro_energies: Dict[str, float] = field(default_factory=dict)
    macro_text: Dict[str, Dict[str, str]] = field(default_factory=dict)
    aixi_schedule_ids: List[int] = field(default_factory=list)
    note: str = "micro→meso→macro white-box atlas parse"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "note": self.note,
            "aixi_schedule_ids": self.aixi_schedule_ids,
            "macro_energies": self.macro_energies,
            "macro_text": self.macro_text,
            "top_areas": [
                {
                    "area_id": a.area_id,
                    "name": a.name,
                    "macro": a.macro,
                    "activation": a.activation,
                    "micro": a.micro,
                    "meso": a.meso,
                    "macro_role": a.macro_role,
                    "aixi_hook": a.aixi_hook,
                    "scheduled": a.scheduled,
                }
                for a in self.top_areas
            ],
        }

    def explain(self, max_areas: int = 12) -> str:
        lines = ["[WhiteBox] 246-area micro→macro parse"]
        lines.append("Macro energies: " + ", ".join(
            f"{k}={v:.3f}" for k, v in sorted(
                self.macro_energies.items(), key=lambda kv: -kv[1]
            )
        ))
        lines.append(f"AIXI scheduled area ids: {self.aixi_schedule_ids[:24]}")
        for a in self.top_areas[:max_areas]:
            flag = "*" if a.scheduled else " "
            lines.append(
                f" {flag}#{a.area_id:03d} [{a.macro}] act={a.activation:.3f}\n"
                f"    micro: {a.micro}\n"
                f"    meso:  {a.meso}\n"
                f"    macro: {a.macro_role}\n"
                f"    AIXI:  {a.aixi_hook}"
            )
        return "\n".join(lines)


class AtlasAnalyzer:
    """Turn activation [B,246] (+ optional schedule mask) into a report."""

    def __init__(self):
        self._macro_of = [c.macro for c in FUNCTION_CARDS]
        assert len(self._macro_of) == NUM_AREAS

    def analyze(
        self,
        activation: torch.Tensor,
        schedule_ids: Optional[Sequence[int]] = None,
        top_k: int = 16,
        batch_index: int = 0,
    ) -> MicroMacroReport:
        if activation.dim() == 1:
            act = activation
        else:
            act = activation[batch_index]
        act = act.detach().float().cpu()
        assert act.numel() == NUM_AREAS

        sched = set(int(i) for i in (schedule_ids or []))
        # area_id is 1..246; schedule may use 0-based or 1-based — normalize
        sched_ids_1based = []
        for i in sched:
            if 0 <= i < NUM_AREAS:
                sched_ids_1based.append(i + 1)
            elif 1 <= i <= NUM_AREAS:
                sched_ids_1based.append(i)
        sched_set = set(sched_ids_1based)

        macro_e = {m: 0.0 for m in MACRO_SYSTEMS}
        macro_n = {m: 0 for m in MACRO_SYSTEMS}
        for i in range(NUM_AREAS):
            m = self._macro_of[i]
            macro_e[m] += float(act[i].item())
            macro_n[m] += 1
        for m in MACRO_SYSTEMS:
            macro_e[m] /= max(1, macro_n[m])

        topk = torch.topk(act, k=min(top_k, NUM_AREAS))
        explains: List[AreaExplain] = []
        for idx, val in zip(topk.indices.tolist(), topk.values.tolist()):
            card = card_for(idx + 1)
            explains.append(
                AreaExplain(
                    area_id=card.area_id,
                    name=card.name,
                    macro=card.macro,
                    activation=float(val),
                    micro=card.micro,
                    meso=card.meso,
                    macro_role=card.macro_role,
                    aixi_hook=card.aixi_hook,
                    scheduled=card.area_id in sched_set,
                )
            )

        return MicroMacroReport(
            top_areas=explains,
            macro_energies=macro_e,
            macro_text=macro_summary(),
            aixi_schedule_ids=sorted(sched_set),
        )
