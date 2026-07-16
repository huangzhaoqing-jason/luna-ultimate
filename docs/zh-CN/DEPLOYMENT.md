# Luna 双轨部署（跨芯片 + Ollama）

## 定案

| 轨道 | 用途 | 命令 |
|------|------|------|
| **Track A — Ollama** | 蒸馏 Llama 拓扑（文本 + 可选 VLM mmproj），便于 `ollama run` | `python3 export/distill_ollama.py` → GGUF → `bash scripts/publish_ollama.sh` |
| **Track B — Native** | 完整 WB-HCA + VLM/VLA/World-Action，满性能本地 | `python3 scripts/run_serve.py --device auto` |

完整 CTM×Mamba×MLA×FlashMoE **不能**原样进入 Ollama 内核。`ollama run luna` = 兼容蒸馏版；VLA/WA = `luna serve`。

## 平台矩阵（诚实）

| 平台 | 后端 | 状态 |
|------|------|------|
| NVIDIA | CUDA / Triton | 就绪（训练+推理主路径） |
| Apple Silicon | MLX / MPS 桥 | 部分就绪；`scripts/probe_backends.py` 探测 |
| 华为昇腾 | CANN / torch_npu | 适配中；不可用时 CPU 回退 |
| Windows (非 N 卡) | DML → PyTorch CPU/ORT 回退 | 冒烟就绪；满性能需 llama.cpp D3D12 |
| CPU x86/ARM | PyTorch | 就绪（`tiny`/`1b`） |

## 能力面（Native）

| 能力 | 说明 |
|------|------|
| LLM | 意义优先解码（Q1B） |
| VLM | V-JEPA → hidden_size 投影后融合 |
| VLA | `ActionHead` 离散+连续动作 |
| WA | `WorldActionModule` 下一状态 + 动作先验 |

所有动作/文本过安全三层门：`charter OR values OR ctm`。

## 快速开始

```bash
# 后端探测
python3 scripts/probe_backends.py

# Native serve（默认端口 11435，避开 Ollama 11434）
python3 scripts/run_serve.py --preset tiny --device cpu --port 11435

# 调用
curl -s http://127.0.0.1:11435/api/tags
curl -s http://127.0.0.1:11435/api/generate \
  -d '{"prompt":"explain gradient descent","max_tokens":16}'

# Ollama 蒸馏（学生权重；GGUF 需本机 llama.cpp）
python3 export/distill_ollama.py --preset tiny --steps 3
bash scripts/publish_ollama.sh export/out/luna-distill
```

## API

- `GET /api/tags` — 模型列表
- `POST /api/generate` — Ollama 风格生成
- `POST /api/chat` / `POST /v1/chat/completions` — 对话（可带 images）
- `POST /api/vla` — 动作预测（安全门控）
