"""Extreme memory thrift for large nominal parameter counts (Pathway 1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class MemoryAccount:
    total_params: int
    active_params: int
    bytes_bf16_full: int
    bytes_resident_estimate: int
    notes: Dict[str, Any]


def count_params(module) -> int:
    return sum(p.numel() for p in module.parameters())


def estimate_resident_memory(
    total_params: int,
    active_fraction: float = 0.05,
    bytes_per_param_resident: float = 0.5,  # INT4-ish
    offload_fraction: float = 0.7,
) -> MemoryAccount:
    """Compare full BF16 residency vs sparse+quant+offload residency."""
    active = int(total_params * active_fraction)
    bf16_full = total_params * 2
    # Only non-offloaded active params stay hot
    hot = int(active * (1.0 - offload_fraction) + total_params * (1.0 - offload_fraction) * 0.01)
    resident = int(hot * bytes_per_param_resident)
    return MemoryAccount(
        total_params=total_params,
        active_params=active,
        bytes_bf16_full=bf16_full,
        bytes_resident_estimate=resident,
        notes={
            "active_fraction": active_fraction,
            "bytes_per_param_resident": bytes_per_param_resident,
            "offload_fraction": offload_fraction,
            "ratio_vs_bf16": resident / max(bf16_full, 1),
        },
    )


def format_account(acc: MemoryAccount) -> str:
    def gb(n: int) -> str:
        return f"{n / (1024**3):.4f} GiB"

    return (
        f"params_total={acc.total_params:,} active={acc.active_params:,}\n"
        f"BF16_full={gb(acc.bytes_bf16_full)}  resident≈{gb(acc.bytes_resident_estimate)}\n"
        f"ratio_vs_bf16={acc.notes['ratio_vs_bf16']:.4f}"
    )
