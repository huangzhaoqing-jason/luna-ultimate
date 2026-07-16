# Luna-Ultimate 全平台部署指南

> **双轨（2026）**：Ollama 蒸馏版（Track A）与 Native `luna serve`（Track B，含 VLM/VLA/World-Action）。
> 详见 [docs/zh-CN/DEPLOYMENT.md](docs/zh-CN/DEPLOYMENT.md)。
> 完整异构架构不能原样进入 Ollama；`ollama run luna` = 兼容学生；满性能/VLA/WA = `python3 scripts/run_serve.py`。

## 架构概览

```
┌──────────────────────────────────────────────────────────────┐
│                    Luna-Ultimate 推理层                       │
│  ┌─────────────┬──────────────┬──────────────┬─────────────┐ │
│  │ Mamba2Scan  │ FlashAttn    │  MoEGate     │ JEPAPredict │ │
│  │    Op       │    Op        │    Op        │    Op       │ │
│  └──────┬──────┴──────┬───────┴──────┬───────┴──────┬──────┘ │
│         │             │              │              │        │
│  ┌──────▼─────────────▼──────────────▼──────────────▼──────┐ │
│  │              RuntimeManager (auto-detect)               │ │
│  └──────┬──────┬──────┬──────┬──────┬──────┬──────────────┘ │
│         │      │      │      │      │      │                │
│    ┌────▼─┐ ┌─▼──┐ ┌─▼──┐ ┌─▼──┐ ┌─▼──┐ ┌─▼────┐          │
│    │Triton│ │MLX │ │CANN│ │DML │ │CPU │ │CUDA  │          │
│    │(CUDA │ │(M1 │ │(Asc│ │(D3D│ │(AVX│ │Graph │          │
│    │ ROCm)│ │ M4)│ │end)│ │12) │ │512)│ │      │          │
│    └──────┘ └────┘ └────┘ └────┘ └────┘ └──────┘          │
└──────────────────────────────────────────────────────────────┘
```

---

## 平台支持矩阵

| 平台 | 硬件 | 加速后端 | 安装方式 | 状态 |
|------|------|----------|----------|------|
| **NVIDIA GPU** | A100/H100/RTX | Triton + CUDA Graph | `USE_CUDA=1 pip install -e .[cuda]` | 就绪 |
| **AMD GPU** | MI300X/7900XTX | Triton + ROCm | `USE_ROCM=1 pip install -e .[rocm]` | 就绪 |
| **Apple Silicon** | M1/M2/M3/M4 | MLX + Metal | `USE_APPLE=1 pip install -e .[apple]` | 就绪 |
| **华为昇腾** | 910B/310P | CANN + torch_npu | `USE_CANN=1 pip install -e .[cann]` | 适配中 |
| **Windows GPU** | AMD/Intel (非N卡) | DirectML + llama.cpp | `USE_DML=1` + CMake 编译 | 适配中 |
| **CPU (x86)** | Intel/AMD | AVX2/AVX-512 | `pip install -e .` | 就绪 |
| **CPU (ARM)** | 鲲鹏/树莓派 | NEON | `pip install -e .` | 就绪 |

---

## 快速开始

### 1. NVIDIA GPU (推荐)

```bash
# 安装 Triton + CUDA 依赖
pip install triton torch flash-attn mamba-ssm

# 验证
python -c "
from runtime_manager import RuntimeManager
rt = RuntimeManager()
# 应输出: Device: NVIDIA xxx | Backend: triton
"
```

### 2. Apple Silicon (M1/M2/M3/M4)

```bash
pip install mlx torch

python -c "
from runtime_manager import RuntimeManager
rt = RuntimeManager()
# 应输出: Device: Apple Silicon | Backend: mlx
"
```

### 3. 华为昇腾

```bash
# 安装 CANN 工具包 (需从华为官网下载)
# 然后安装 torch_npu
pip install torch-npu

python -c "
from runtime_manager import RuntimeManager
rt = RuntimeManager()
# 应输出: Device: Huawei Ascend | Backend: cann
"
```

### 4. CPU (通用)

```bash
pip install -e .

python -c "
from runtime_manager import RuntimeManager
rt = RuntimeManager(device_type='cpu')
# 应输出: Device: xxx | Backend: pytorch
"
```

---

## 算子映射表

| Luna 算子 | CUDA/ROCm | Apple MLX | Huawei CANN | DirectML | CPU |
|-----------|-----------|-----------|-------------|----------|-----|
| Mamba2Scan | Triton `_mamba2_scan_kernel` | `mlx.fast.ssd` | Ascend C `npu_mamba2_scan` | GGML D3D12 | AVX2 向量化 |
| FlashAttention | Triton `_flash_attention_fwd_kernel` | `mlx.fast.sdpa` | `npu_fusion_attention` | GGML D3D12 | 标准 PyTorch |
| MoEGate | Triton `_moe_topk_kernel` | `mlx.topk` | `torch.topk` (NPU) | GGML topk | 标准 PyTorch |
| JEPAPredict | PyTorch Linear | MLX Linear | Ascend MatMul | GGML MatMul | 标准 PyTorch |

---

## Triton 内核编译

Triton 一次编写，多平台编译：

```
Triton Python DSL
    │
    ├──→ triton.compile() → PTX (NVIDIA CUDA)
    ├──→ triton.compile() → AMDGCN (AMD ROCm)
    ├──→ triton.compile() → SPIR-V (Intel GPU, 实验性)
    └──→ triton.compile() → MLIR → CANN (华为昇腾, 实验性)
```

### 关键 Triton 内核

| 文件 | 内核 | 加速比 |
|------|------|--------|
| `triton_kernels.py` | `_mamba2_scan_kernel` | 3-5× vs PyTorch |
| `triton_kernels.py` | `_flash_attention_fwd_kernel` | 2-4× vs PyTorch |
| `triton_kernels.py` | `_moe_topk_kernel` | 1.5-2× vs PyTorch |

---

## CUDA Graph 捕获

CUDA Graph 是消除 Python 开销、实现极限推理速度的关键：

```python
from triton_kernels import CUDAGraphWrapper

wrapper = CUDAGraphWrapper()
wrapper.capture(model.forward, example_input)

# 推理循环：零 Python 开销
for token in range(max_tokens):
    output = wrapper.replay(input_tensor)  # ~30% faster
```

---

## 编译构建

```bash
# C++ 内核编译 (可选，用于自定义算子)
mkdir build && cd build
cmake .. -DUSE_CUDA=ON -DUSE_TRITON=ON
cmake --build . -j$(nproc)

# 纯 Python 安装 (推荐)
pip install -e .
```

---

## 依赖清单

```
# 核心
torch>=2.0.0
numpy>=1.24.0

# NVIDIA/AMD GPU
triton>=2.1.0
flash-attn>=2.5.0
mamba-ssm>=2.0.0

# Apple Silicon
mlx>=0.8.0

# 华为昇腾
torch-npu

# Windows DirectML
llama-cpp-python

# 训练
deepspeed>=0.12.0
bitsandbytes>=0.41.0
transformers>=4.35.0
wandb>=0.15.0
```