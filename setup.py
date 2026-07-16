"""Luna-Ultimate: Cross-Platform Setup Configuration.

Supports dynamic compilation of backend kernels based on environment:
  USE_CUDA=1     → NVIDIA GPU (Triton + CUDA kernels)
  USE_ROCM=1     → AMD GPU (Triton + ROCm kernels)
  USE_APPLE=1    → Apple Silicon (MLX backend)
  USE_CANN=1     → Huawei Ascend (CANN backend)
  USE_DML=1      → Windows DirectML (llama.cpp GGML)
  USE_CPU=1      → CPU-only (vectorized fallback)

Usage:
  pip install -e .                    # Auto-detect
  USE_CUDA=1 pip install -e .         # Force CUDA
  USE_APPLE=1 pip install -e .[apple]  # Apple Silicon with MLX extras
"""

from setuptools import setup, find_packages
import os

# ==================== Dependency Groups ====================

install_requires = [
    "torch>=2.0.0",
    "numpy>=1.24.0",
    "einops>=0.7.0",
]

extras_require = {
    # NVIDIA GPU
    "cuda": [
        "triton>=2.1.0",
        "flash-attn>=2.5.0",
        "mamba-ssm>=2.0.0",
    ],

    # AMD GPU (ROCm)
    "rocm": [
        "triton>=2.1.0",
    ],

    # Apple Silicon
    "apple": [
        "mlx>=0.8.0",
    ],

    # Huawei Ascend NPU
    "cann": [
        "torch-npu",
    ],

    # Windows DirectML
    "dml": [
        "llama-cpp-python",
    ],

    # Training dependencies
    "train": [
        "deepspeed>=0.12.0",
        "bitsandbytes>=0.41.0",
        "transformers>=4.35.0",
        "datasets>=2.14.0",
        "tokenizers>=0.15.0",
        "wandb>=0.15.0",
    ],

    # Full install (all platforms)
    "all": [
        "triton>=2.1.0",
        "flash-attn>=2.5.0",
        "mamba-ssm>=2.0.0",
        "mlx>=0.8.0",
        "deepspeed>=0.12.0",
        "bitsandbytes>=0.41.0",
        "transformers>=4.35.0",
        "wandb>=0.15.0",
    ],
}

# ==================== Auto-detect Backend ====================

def auto_detect_backend():
    """Auto-detect the best available backend for installation."""
    import platform
    import sys

    # Check environment variables
    if os.environ.get("USE_CPU"):
        return "cpu"

    if os.environ.get("USE_CANN"):
        return "cann"

    if os.environ.get("USE_APPLE"):
        return "apple"

    if platform.system() == "Darwin" and "arm" in platform.machine():
        return "apple"

    if os.environ.get("USE_ROCM"):
        return "rocm"

    if os.environ.get("USE_CUDA"):
        return "cuda"

    if os.environ.get("USE_DML"):
        return "dml"

    # Try CUDA detection
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            if "AMD" in device_name:
                return "rocm"
            return "cuda"
    except ImportError:
        pass

    return "cpu"


# ==================== Setup ====================

backend = auto_detect_backend()
print(f"[Luna-Ultimate Setup] Detected backend: {backend}")
print(f"  Use environment variables to override: USE_CUDA=1, USE_APPLE=1, etc.")

setup(
    name="luna-ultimate",
    version="0.1.0",
    description="Luna-Ultimate: 550B Hybrid Architecture Model (CTM x Mamba2-SSD x MLA x FlashMoE)",
    author="huangzhaoqing-jason",
    author_email="luna-ultimate@github.com",
    url="https://github.com/huangzhaoqing-jason/luna-ultimate",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=install_requires,
    extras_require=extras_require,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "luna-inference=luna_ultimate.inference_demo:main",
            "luna-train=luna_ultimate.train:main",
            "luna-runtime=luna_ultimate.runtime_manager:main",
        ],
    },
)