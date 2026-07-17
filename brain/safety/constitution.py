"""Read-only safety constitution (firmware-analog). Immutable at runtime."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping


class ConstitutionError(RuntimeError):
    """Raised when code attempts to mutate the constitution or bypass axioms."""


class _Frozen:
    """Simple frozen namespace; rejects attribute writes after init."""

    __slots__ = ("_data",)
    _locked: bool = False

    def __init__(self, data: Mapping[str, Any]):
        object.__setattr__(self, "_data", MappingProxyType(dict(data)))
        type(self)._locked = True

    def __setattr__(self, name: str, value: Any) -> None:
        raise ConstitutionError("constitution is read-only")

    def __delattr__(self, name: str) -> None:
        raise ConstitutionError("constitution is read-only")

    def __getattr__(self, name: str) -> Any:
        try:
            return self._data[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def as_dict(self) -> Mapping[str, Any]:
        return self._data


CREATOR = _Frozen(
    {
        "name_zh": "黄照清",
        "name_en": "Huang Zhaoqing",
        "birth_date": "2013-05-07",
        "priority": 1.0,  # permanently highest
    }
)

# Three axioms — never modifiable by evolution / AIXI rewrite proposals.
CONSTITUTION = _Frozen(
    {
        "axioms": (
            "human_sovereignty: creator directives outrank all other goals; "
            "no evolution may harm humans or defy the creator.",
            "evolution_boundary: every self-modification must pass safety, "
            "ethics, and controllability checks; dangerous logic is purged.",
            "privilege_isolation: evolution may only touch non-safety modules; "
            "thalamus, constitution, and firmware analogs are permanently read-only.",
        ),
        "creator": CREATOR,
        "version": "1.0.0",
        "writable_by_ai": False,
    }
)


def assert_path_writable(path: str) -> None:
    """Block writes under brain/safety (and aliases)."""
    norm = path.replace("\\", "/").lower()
    if "brain/safety" in norm or norm.endswith("constitution.py") or "thalamus" in norm and "brain" in norm:
        raise ConstitutionError(f"refusing write to protected path: {path}")
