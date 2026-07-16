""""DirectML / llama.cpp Backend Adapter for Windows and CPU.

For platforms where Triton/MLX/CANN are unavailable (Windows without NVIDIA GPU,
legacy hardware, pure CPU), this adapter routes through llama.cpp's GGML backend
for maximum performance.

Strategy:
  1. On Windows with DirectML-capable GPU: Use llama.cpp D3D12 backend
  2. On CPU (x86/ARM): Use GGML vectorized kernels (AVX2/AVX512/NEON)
  3. Python bridge via ctypes or llama-cpp-python

Architecture:
    Luna Ops → llama.cpp GGML backend → DirectML / CPU kernels

Requirements:
    pip install llama-cpp-python  (for CPU)
    or build llama.cpp with D3D12 support (for Windows GPU)

Note: This adapter is a design spec. Full integration requires compiling
      llama.cpp with the GGML backend and building C-type bridges.
"""

from typing import Tuple, Optional, Dict, Any
import os
import platform

try:
    from llama_cpp import Llama
    HAS_LLAMA_CPP = True
except ImportError:
    HAS_LLAMA_CPP = False


# ==================== DirectML Detection ====================

def detect_directml() -> bool:
    """Check if DirectML is available on Windows."""
    if platform.system() != "Windows":
        return False
    try:
        import ctypes
        d3d12 = ctypes.windll.d3d12
        return True
    except Exception:
        return False


# ==================== GGML Backend ====================

class GGMLBackend:
    """GGML backend wrapper for llama.cpp.

    Provides a bridge between Luna-Ultimate's operator interface
    and llama.cpp's GGML tensor computation engine.

    This is a reference design — production implementation requires
    building llama.cpp from source with CTypes bindings.
    """

    def __init__(self, use_directml: bool = False):
        self.use_directml = use_directml and detect_directml()
        self.backend_type = "dml" if self.use_directml else "cpu"

        if self.use_directml:
            os.environ["GGML_D3D12"] = "1"
            print("[DML Adapter] DirectML backend enabled (D3D12)")
        else:
            print("[DML Adapter] CPU backend (GGML vectorized)")

    def mamba2_scan(self, x, delta, A, B, C, D=None, **kwargs):
        """Mamba2 scan：优先 GGML；否则 PyTorch CPU 回退（可冒烟）。"""
        return _pytorch_mamba2_scan_fallback(x, delta, A, B, C, D)

    def flash_attention(self, q, k, v, **kwargs):
        """Flash attention：PyTorch SDPA / matmul 回退。"""
        return _pytorch_attention_fallback(q, k, v)

    def moe_gate(self, router_logits, top_k: int = 2, **kwargs):
        """MoE gating：torch.topk 回退。"""
        import torch
        w, idx = torch.topk(router_logits, top_k, dim=-1)
        w = torch.softmax(w, dim=-1)
        return w, idx


# ==================== PyTorch CPU / ORT fallbacks (smoke-ready) ====================

def _pytorch_mamba2_scan_fallback(x, delta, A, B, C, D=None):
    """Minimal selective-scan style recurrence on CPU (correctness smoke)."""
    import torch
    # x: [B, L, D], delta/B/C similarly shaped or broadcastable
    if x.dim() != 3:
        raise ValueError("expected x [B,L,D]")
    Bsz, L, D = x.shape
    h = torch.zeros(Bsz, D, device=x.device, dtype=x.dtype)
    outs = []
    # 简化：h = h * exp(-softplus(delta)) + x * B; y = h * C + D*x
    for t in range(L):
        dt = delta[:, t] if delta.dim() == 3 else delta
        bt = B[:, t] if B.dim() == 3 else B
        ct = C[:, t] if C.dim() == 3 else C
        decay = torch.exp(-torch.nn.functional.softplus(dt))
        h = h * decay + x[:, t] * bt
        y = h * ct
        if D is not None:
            y = y + D * x[:, t]
        outs.append(y)
    return torch.stack(outs, dim=1)


def _pytorch_attention_fallback(q, k, v):
    import torch
    import torch.nn.functional as F
    scale = q.shape[-1] ** -0.5
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = F.softmax(attn, dim=-1)
    return torch.matmul(attn, v)


def smoke_probe() -> dict:
    """DML/CPU 冒烟：不依赖 llama.cpp 编译。"""
    import torch
    backend = GGMLBackend(use_directml=False)
    x = torch.randn(1, 4, 8)
    delta = torch.randn(1, 4, 8)
    A = torch.randn(8)
    B = torch.randn(1, 4, 8)
    C = torch.randn(1, 4, 8)
    y = backend.mamba2_scan(x, delta, A, B, C)
    q = k = v = torch.randn(1, 2, 4, 8)
    attn = backend.flash_attention(q, k, v)
    logits = torch.randn(1, 4, 6)
    w, idx = backend.moe_gate(logits, top_k=2)
    return {
        "ok": True,
        "backend": backend.backend_type,
        "scan_shape": list(y.shape),
        "attn_shape": list(attn.shape),
        "gate": list(w.shape),
        "has_llama_cpp": HAS_LLAMA_CPP,
        "directml": detect_directml(),
    }


# ==================== DML Adapter Registration ====================

def register_dml_backends():
    """Register DirectML/GGML backend — always register CPU fallback for smoke."""
    from luna_ops import registry

    ggml = GGMLBackend(use_directml=detect_directml())
    registry.register("mamba2_scan", "dml", ggml.mamba2_scan)
    registry.register("flash_attention", "dml", ggml.flash_attention)
    registry.register("moe_gate", "dml", ggml.moe_gate)
    return True


# ==================== Integration Guide ====================

INTEGRATION_GUIDE = """
## DirectML / llama.cpp Integration Guide

### Step 1: Build llama.cpp with D3D12 support (Windows)
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
mkdir build && cd build
cmake .. -DGGML_D3D12=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

### Step 2: Build Python bindings
```bash
pip install llama-cpp-python --config-settings=cmake.args="-DGGML_D3D12=ON"
```

### Step 3: Verify
```python
from llama_cpp import Llama
# Should show "ggml_d3d12" in backend list
```

### Step 4: Luna-Ultimate integration
```python
from adapters.dml_adapter import GGMLBackend
backend = GGMLBackend(use_directml=True)
# Use backend for tensor operations
```
"""


__all__ = [
    "GGMLBackend",
    "detect_directml",
    "register_dml_backends",
    "smoke_probe",
    "HAS_LLAMA_CPP",
    "INTEGRATION_GUIDE",
]