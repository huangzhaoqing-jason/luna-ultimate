"""Luna-Ultimate Cross-Platform Runtime Manager.

Auto-detects available hardware and loads the optimal backend for each operator.
Provides a unified interface that dispatches to:
  - Triton kernels (CUDA/ROCm/CANN partial)
  - MLX bridge (Apple Silicon)
  - CANN adapter (Huawei Ascend NPU)
  - DirectML (Windows GPUs)
  - CPU vectorized fallback (x86/ARM)

Usage:
    from runtime_manager import RuntimeManager
    rt = RuntimeManager()
    output = rt.mamba2_scan(x, delta, A, B, C, D)
"""

import torch
import os
import sys
import platform
import subprocess
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field


# ==================== Hardware Detection ====================

@dataclass
class DeviceInfo:
    """Detected hardware capabilities."""
    device_type: str = "cpu"
    device_name: str = "Unknown"
    total_memory_gb: float = 0.0
    compute_capability: Tuple[int, int] = (0, 0)
    supports_triton: bool = False
    supports_mlx: bool = False
    supports_cann: bool = False
    supports_dml: bool = False
    supports_cuda_graph: bool = False
    supports_bf16: bool = False
    num_devices: int = 0


def detect_nvidia() -> Optional[DeviceInfo]:
    """Detect NVIDIA GPU capabilities."""
    if not torch.cuda.is_available():
        return None

    props = torch.cuda.get_device_properties(0)
    info = DeviceInfo(
        device_type="cuda",
        device_name=props.name,
        total_memory_gb=props.total_mem / (1024 ** 3),
        compute_capability=(props.major, props.minor),
        supports_bf16=props.major >= 8,  # Ampere+
        supports_cuda_graph=True,
        num_devices=torch.cuda.device_count(),
    )

    # Check Triton support
    try:
        import triton
        info.supports_triton = True
    except ImportError:
        pass

    return info


def detect_amd() -> Optional[DeviceInfo]:
    """Detect AMD GPU (ROCm)."""
    try:
        if torch.cuda.is_available() and "AMD" in torch.cuda.get_device_name(0):
            props = torch.cuda.get_device_properties(0)
            info = DeviceInfo(
                device_type="rocm",
                device_name=props.name,
                total_memory_gb=props.total_mem / (1024 ** 3),
                supports_bf16=False,
                num_devices=torch.cuda.device_count(),
            )
            try:
                import triton
                info.supports_triton = True
            except ImportError:
                pass
            return info
    except Exception:
        pass
    return None


def detect_apple() -> Optional[DeviceInfo]:
    """Detect Apple Silicon (M1/M2/M3/M4)."""
    if platform.system() != "Darwin":
        return None

    is_apple_silicon = False
    try:
        result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                capture_output=True, text=True)
        if "Apple" in result.stdout:
            is_apple_silicon = True
    except Exception:
        pass

    if not is_apple_silicon:
        return None

    # Check MPS availability
    if not torch.backends.mps.is_available():
        return None

    info = DeviceInfo(
        device_type="mps",
        device_name="Apple Silicon",
        total_memory_gb=0.0,  # Unified memory
        supports_bf16=False,
        num_devices=1,
    )

    # Check MLX
    try:
        import mlx.core as mx
        info.supports_mlx = True
    except ImportError:
        pass

    return info


def detect_huawei() -> Optional[DeviceInfo]:
    """Detect Huawei Ascend NPU."""
    try:
        import torch_npu
        if torch_npu.npu.is_available():
            info = DeviceInfo(
                device_type="npu",
                device_name="Huawei Ascend",
                total_memory_gb=0.0,
                supports_bf16=True,
                supports_cann=True,
                num_devices=torch_npu.npu.device_count(),
            )
            # Triton may support CANN via torch_npu bridge
            try:
                import triton
                info.supports_triton = True
            except ImportError:
                pass
            return info
    except ImportError:
        pass
    return None


def detect_cpu() -> DeviceInfo:
    """Detect CPU capabilities."""
    import multiprocessing as mp

    info = DeviceInfo(
        device_type="cpu",
        device_name=platform.processor() or "Unknown",
        total_memory_gb=0.0,
        num_devices=mp.cpu_count(),
    )

    # Check for AVX2/AVX512
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                flags = f.read()
                if "avx512" in flags:
                    info.device_name += " (AVX-512)"
                elif "avx2" in flags:
                    info.device_name += " (AVX2)"
        except Exception:
            pass

    return info


# ==================== Runtime Manager ====================

class RuntimeManager:
    """Unified runtime for cross-platform Luna-Ultimate inference.

    Auto-detects hardware and loads optimal operator implementations.
    Provides a single API for all backends.

    Usage:
        rt = RuntimeManager(device_type="auto")
        rt.print_info()

        # Direct operator access
        y, state = rt.mamba2_scan(x, delta, A, B, C, D)
        attn_out = rt.flash_attention(q, k, v, scale, causal=True)
        weights, indices, counts = rt.moe_gate(logits, top_k=4)
    """

    def __init__(self, device_type: str = "auto", verbose: bool = True):
        self.verbose = verbose
        self.device_info: DeviceInfo = None
        self.backend_name: str = "pytorch"

        # Detect hardware
        if device_type == "auto":
            self.device_info = self._auto_detect()
        elif device_type == "cuda":
            self.device_info = detect_nvidia() or detect_cpu()
        elif device_type == "mps":
            self.device_info = detect_apple() or detect_cpu()
        elif device_type == "npu":
            self.device_info = detect_huawei() or detect_cpu()
        elif device_type == "cpu":
            self.device_info = detect_cpu()
        else:
            self.device_info = detect_cpu()

        self.device_type = self.device_info.device_type

        # Load appropriate backends
        self._load_backends()

        if self.verbose:
            self.print_info()

    def _auto_detect(self) -> DeviceInfo:
        """Auto-detect best available hardware."""
        # Priority: CUDA > ROCm > Apple > Huawei > CPU
        for detector in [detect_nvidia, detect_amd, detect_apple, detect_huawei]:
            result = detector()
            if result is not None:
                return result
        return detect_cpu()

    def _load_backends(self):
        """Load optimal backend implementations for current hardware."""
        from luna_ops import registry, set_backend

        if self.device_info.supports_triton:
            self.backend_name = "triton"
            try:
                from triton_kernels import register_triton_backends
                ok = register_triton_backends()
                if ok and self.verbose:
                    print("[Runtime] Triton kernels loaded (CUDA/ROCm)")
            except ImportError:
                pass

        if self.device_info.supports_mlx:
            self.backend_name = "mlx"
            try:
                from adapters.mlx_adapter import register_mlx_backends
                register_mlx_backends()
                if self.verbose:
                    print("[Runtime] MLX backend loaded (Apple Silicon)")
            except ImportError:
                pass

        if self.device_info.supports_cann:
            self.backend_name = "cann"
            try:
                from adapters.cann_adapter import register_cann_backends
                register_cann_backends()
                if self.verbose:
                    print("[Runtime] CANN backend loaded (Huawei Ascend)")
            except ImportError:
                pass

        set_backend(self.backend_name)

    def print_info(self):
        """Print detected hardware and backend information."""
        info = self.device_info
        print("=" * 60)
        print("  Luna-Ultimate Runtime Manager")
        print("=" * 60)
        print(f"  Device:       {info.device_name}")
        print(f"  Type:         {info.device_type}")
        print(f"  Backend:       {self.backend_name}")
        print(f"  Triton:        {'YES' if info.supports_triton else 'no'}")
        print(f"  MLX:           {'YES' if info.supports_mlx else 'no'}")
        print(f"  CANN:          {'YES' if info.supports_cann else 'no'}")
        print(f"  CUDA Graph:    {'YES' if info.supports_cuda_graph else 'no'}")
        print(f"  BF16:          {'YES' if info.supports_bf16 else 'no'}")
        print(f"  Devices:       {info.num_devices}")
        if info.total_memory_gb > 0:
            print(f"  Memory:        {info.total_memory_gb:.1f} GB")
        print("=" * 60)

    def mamba2_scan(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B_ssm: torch.Tensor,
        C_ssm: torch.Tensor,
        D: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mamba2 selective scan."""
        from luna_ops import Mamba2ScanOp
        return Mamba2ScanOp.apply(x, delta, A, B_ssm, C_ssm, D, True)

    def flash_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        softmax_scale: float,
        causal: bool = True,
    ) -> torch.Tensor:
        """Flash Attention for MLA."""
        from luna_ops import FlashAttentionOp
        return FlashAttentionOp.apply(q, k, v, softmax_scale, causal, True)

    def moe_gate(
        self,
        router_logits: torch.Tensor,
        top_k: int = 4,
        capacity_factor: float = 1.25,
        num_experts: int = 48,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """MoE Top-K gating."""
        from luna_ops import MoEGateOp
        return MoEGateOp.apply(router_logits, top_k, capacity_factor, num_experts, True)

    def jepa_predict(
        self,
        context_features: torch.Tensor,
        mask: torch.Tensor,
        ids_restore: torch.Tensor,
        predictor_weight: torch.Tensor,
        predictor_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """JEPA feature prediction."""
        from luna_ops import JEPAPredictorOp
        return JEPAPredictorOp.apply(
            context_features, mask, ids_restore,
            predictor_weight, predictor_bias, True,
        )

    def get_optimal_dtype(self) -> torch.dtype:
        """Get the optimal compute dtype for current hardware."""
        if self.device_info.supports_bf16:
            return torch.bfloat16
        return torch.float16

    def get_device(self) -> torch.device:
        """Get the torch device for current hardware."""
        mapping = {
            "cuda": "cuda:0",
            "rocm": "cuda:0",
            "mps": "mps",
            "npu": "npu:0",
            "cpu": "cpu",
        }
        return torch.device(mapping.get(self.device_type, "cpu"))


# ==================== Singleton Access ====================

_runtime_instance: Optional[RuntimeManager] = None


def get_runtime(device_type: str = "auto", verbose: bool = True) -> RuntimeManager:
    """Get or create the global RuntimeManager instance."""
    global _runtime_instance
    if _runtime_instance is None:
        _runtime_instance = RuntimeManager(device_type=device_type, verbose=verbose)
    return _runtime_instance


# ==================== CLI ====================

if __name__ == "__main__":
    rt = RuntimeManager()
    print(f"\nOptimal dtype: {rt.get_optimal_dtype()}")
    print(f"Device: {rt.get_device()}")