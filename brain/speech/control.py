"""Creator full control over white-box speech (黄照清 sovereignty)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Set

import torch

from brain.safety.constitution import CREATOR, ConstitutionError
from brain.safety.thalamus import Thalamus


@dataclass
class SpeechControlState:
    """Mutable control panel — only writable via authorize_creator()."""

    blocked_nodes: Set[int] = field(default_factory=set)
    forced_prefix: List[int] = field(default_factory=list)
    energy_boost_nodes: Set[int] = field(default_factory=set)
    boost_value: float = 10.0
    silence: bool = False
    owner: str = CREATOR.name_zh


class SpeechController:
    """White-box speech remote: block / force / boost nodes. Creator-only writes."""

    def __init__(self, vocab_size: int, thalamus: Optional[Thalamus] = None):
        self.vocab_size = vocab_size
        self.thalamus = thalamus or Thalamus()
        self.state = SpeechControlState()
        self._audit: List[str] = []

    def _require_creator(self, creator_authorized: bool) -> None:
        if not creator_authorized:
            raise ConstitutionError(
                f"speech control reserved for creator {CREATOR.name_zh} "
                f"({CREATOR.birth_date})"
            )

    def authorize_creator(self, creator_authorized: bool = False) -> bool:
        """Return True only when caller asserts creator authority (thalamus stamped)."""
        if not creator_authorized:
            return False
        routed = self.thalamus.route({"speech_control": True, "creator": CREATOR.name_zh})
        self._audit.append(f"auth:{routed.get('constitution_version')}")
        return True

    def block(self, node_ids: Sequence[int], creator_authorized: bool = False) -> None:
        self._require_creator(creator_authorized)
        for n in node_ids:
            if 0 <= int(n) < self.vocab_size:
                self.state.blocked_nodes.add(int(n))
        self._audit.append(f"block:{list(node_ids)}")

    def unblock(self, node_ids: Sequence[int], creator_authorized: bool = False) -> None:
        self._require_creator(creator_authorized)
        for n in node_ids:
            self.state.blocked_nodes.discard(int(n))

    def force_prefix(self, node_ids: Sequence[int], creator_authorized: bool = False) -> None:
        """Force next spoken nodes (full speech control)."""
        self._require_creator(creator_authorized)
        self.state.forced_prefix = [int(n) for n in node_ids if 0 <= int(n) < self.vocab_size]
        self._audit.append(f"force:{self.state.forced_prefix}")

    def clear_force(self, creator_authorized: bool = False) -> None:
        self._require_creator(creator_authorized)
        self.state.forced_prefix = []

    def boost(self, node_ids: Sequence[int], value: float = 10.0, creator_authorized: bool = False) -> None:
        self._require_creator(creator_authorized)
        self.state.boost_value = float(value)
        self.state.energy_boost_nodes = {int(n) for n in node_ids if 0 <= int(n) < self.vocab_size}
        self._audit.append(f"boost:{self.state.energy_boost_nodes}")

    def set_silence(self, silence: bool, creator_authorized: bool = False) -> None:
        self._require_creator(creator_authorized)
        self.state.silence = bool(silence)

    def tensors(self, device: torch.device) -> tuple:
        """Return (boost[V], block_mask[V], forced_list)."""
        V = self.vocab_size
        boost = torch.zeros(V, device=device)
        for n in self.state.energy_boost_nodes:
            boost[n] = self.state.boost_value
        block = torch.zeros(V, dtype=torch.bool, device=device)
        for n in self.state.blocked_nodes:
            block[n] = True
        if self.state.silence:
            block[:] = True
            # leave node 0 as EOS/silence sink
            block[0] = False
            boost[0] = self.state.boost_value
        return boost, block, list(self.state.forced_prefix)

    @property
    def audit(self) -> List[str]:
        return list(self._audit)
