"""设备选型：封装 RuntimeManager，支持 auto|cuda|mlx|cann|dml|cpu。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch

DeviceName = Literal["auto", "cuda", "mlx", "cann", "dml", "cpu"]


@dataclass
class DeviceChoice:
    name: str
    torch_device: torch.device
    backend: str
    detail: str


def pick_device(preference: str = "auto") -> DeviceChoice:
    pref = (preference or "auto").lower().strip()
    info = None
    try:
        from runtime_manager import RuntimeManager
        rt = RuntimeManager()
        info = getattr(rt, "device_info", None) or getattr(rt, "info", None)
        backend = getattr(rt, "backend_name", None) or getattr(rt, "active_backend", "pytorch")
        if callable(backend):
            backend = backend()
    except Exception as e:
        backend = f"pytorch_fallback:{e}"

    if pref == "cuda" and torch.cuda.is_available():
        return DeviceChoice("cuda", torch.device("cuda"), "triton/cuda", "explicit cuda")
    if pref == "cpu":
        return DeviceChoice("cpu", torch.device("cpu"), "cpu", "explicit cpu")
    if pref == "mlx":
        # MLX 权重仍经 torch CPU/MPS 桥接跑模型；标记后端名
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return DeviceChoice("mlx", torch.device("mps"), "mlx/mps", "apple mps bridge")
        return DeviceChoice("mlx", torch.device("cpu"), "mlx/cpu", "mlx requested; cpu torch bridge")
    if pref == "cann":
        try:
            import torch_npu  # noqa: F401
            return DeviceChoice("cann", torch.device("npu:0"), "cann", "huawei ascend")
        except Exception:
            return DeviceChoice("cann", torch.device("cpu"), "cann/cpu", "cann unavailable; cpu fallback")
    if pref == "dml":
        # DirectML：无专用 torch 设备时回退 CPU（适配器冒烟路径）
        return DeviceChoice("dml", torch.device("cpu"), "dml/cpu", "dml via cpu/ort fallback")

    # auto
    if torch.cuda.is_available():
        return DeviceChoice("cuda", torch.device("cuda"), str(backend), "auto→cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return DeviceChoice("mlx", torch.device("mps"), "mlx/mps", "auto→mps")
    return DeviceChoice("cpu", torch.device("cpu"), "cpu", f"auto→cpu ({backend})")
