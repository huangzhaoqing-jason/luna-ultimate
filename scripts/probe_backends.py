#!/usr/bin/env python3
"""探测 Apple/Windows/Huawei/NVIDIA/CPU 适配器冒烟状态。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    from serve.device_router import pick_device
    from adapters import dml_adapter, mlx_adapter, cann_adapter

    report = {
        "device_auto": pick_device("auto").__dict__,
        "device_cpu": pick_device("cpu").__dict__,
        "device_dml": pick_device("dml").__dict__,
        "dml": dml_adapter.smoke_probe(),
        "mlx": mlx_adapter.smoke_probe(),
        "cann": cann_adapter.smoke_probe(),
    }
    # torch device 不可 JSON 序列化
    for k in list(report):
        if k.startswith("device_") and hasattr(report[k].get("torch_device"), "type"):
            report[k]["torch_device"] = str(report[k]["torch_device"])
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
