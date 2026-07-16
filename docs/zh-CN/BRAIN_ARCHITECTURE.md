# Luna 全脑异构认知架构（WB-HCA）

## 目标

在**最低本地硬件成本**下逼近顶级能力：按脑区映射模块，丘脑按需唤醒，纯意义优先解码（Q1B，无自回归 fallback）。

## 脑区 → 模块

| 脑区 | 功能 | Luna 模块 |
|---|---|---|
| 前额叶 | 规划 / 审慎 | `CTM` + `MeaningPlanner` |
| 丘脑 | 动态路由 GWT | `ThalamusRouter` |
| 顶叶 | 数学 / 符号 | `ParietalReasoner` |
| 小脑 | 习惯 / 缓存 | `CerebellarCorrector` + Mamba2 |
| 海马体 | 情景记忆 / LoRA | `EpisodicMemory` + LoRAHook |
| 颞叶 | 知识 | FlashMoE + `rag/` |
| 语言区 | 解码 | `MeaningFirstDecoder`（无 AR fallback） |
| 脑干 / 杏仁核 | 安全门 | charter + values + SafetyCTM |

完整表见 `modeling_neuroarch.py::WB_HCA`。

## 意义优先解码（Q1B / Q3B）

1. `MeaningPlanner`：prompt → 目标语义 latent  
2. `MeaningFirstDecoder`：纯按语义解码 token  
3. `CollapseDetector`：坍塌**只标记、不回退**（tiny 上可读输出差是预期）

## 算力阶梯

| Preset | 硬件 | 用途 |
|---|---|---|
| `tiny` | CPU / 8GB | 开发、evolve 冒烟、安全/价值观测试 |
| `1b` | 24GB | 本机端到端 |
| `7b` | 1–8×80GB | 能力冲刺 |
| `550b` | 多机 | 挑战档（基因组稳定后再上） |

## 命令

```bash
python scripts/run_neuroarch.py --task-text "证明这个定理"
python evolve_ci.py --n-runs 3 --preset tiny
```
