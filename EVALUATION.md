# CTM Validation Experiment Plan

## 1. 实验目标

验证 Continuous Thought Module (CTM) 在 Luna-Ultimate 550B 模型中的三个核心能力：

| 能力 | 描述 | 验证方法 |
|------|------|----------|
| **推理深度自适应** | 模型能否根据输入复杂度自动选择 1-4 ticks | 同步矩阵熵 vs 任务难度相关性分析 |
| **推理质量增益** | CTM 多 tick 是否真正提升复杂推理质量 | A/B 对比 (CTM=OFF vs CTM=ON) |
| **计算效率** | 自适应早退是否有效节省计算 | 平均 ticks 统计 + 吞吐量对比 |

---

## 2. 实验设计

### 2.1 数据集选择

| 数据集 | 任务类型 | 预期难度 | 预期 ticks | 样本数 |
|--------|----------|----------|------------|--------|
| GSM8K | 小学数学推理 | 中 | 2-3 | 1319 |
| MATH | 竞赛级数学 | 高 | 3-4 | 5000 |
| HumanEval | 代码生成 | 中-高 | 2-4 | 164 |
| MMLU (STEM) | 知识问答 | 低-中 | 1-2 | ~3000 |
| BBH (BIG-Bench Hard) | 复杂逻辑推理 | 高 | 3-4 | 6511 |
| SimpleQA | 简单事实问答 | 低 | 1 | 4326 |
| GSM-Symbolic | 符号推理变体 | 中-高 | 2-4 | 5000 |

### 2.2 对比组设计

```
Group A: CTM=OFF (baseline)
  - 模型不加载 CTM 模块
  - 或 CTM 固定使用 1 tick 且不注入残差

Group B: CTM=ON, ticks=1 (固定)
  - CTM 始终运行 1 tick
  - 测试最低计算量下的推理能力

Group C: CTM=ON, ticks=2 (固定)
  - CTM 始终运行 2 ticks

Group D: CTM=ON, ticks=4 (固定)
  - CTM 始终运行 4 ticks (全深度推理)
  - 预期最高推理质量

Group E: CTM=ON, adaptive (自适应)
  - CTM 根据同步矩阵熵动态选择 1-4 ticks
  - 目标：在 Group D 质量的 95% 以上，同时节省 30%+ 计算量
```

### 2.3 评价指标

| 指标 | 计算方式 | 目标 |
|------|----------|------|
| **Accuracy / Pass@1** | 标准评测指标 | E 组 ≈ D 组 (95%+) |
| **平均 ticks** | 所有样本 ticks 的均值 | E 组 < 2.8 (节省 30%+) |
| **准确率-效率比** | Accuracy / avg_ticks | E 组最高 |
| **同步矩阵熵** | Von Neumann entropy of sync matrix | 与任务难度正相关 |
| **早退准确率** | 1-tick 样本的准确率 | 不应低于 4-tick 的 90% |
| **tick 分布** | 1/2/3/4 tick 的比例 | 呈现递减分布 |

---

## 3. 实施步骤

### Phase 1: 独立 CTM 微调 (1-2 天)

```python
# 冻结主模型，仅训练 CTM 模块
for name, param in model.named_parameters():
    if "ctm" not in name:
        param.requires_grad = False

# 使用推理任务数据集训练 CTM 的早退决策
# Loss = task_loss + 0.01 * tick_penalty
optimizer = AdamW(model.ctm.parameters(), lr=1e-4)
```

**验证点**: CTM 能否在简单问题上稳定选择 1-2 ticks

### Phase 2: 全模型联合训练 (3-5 天)

```python
# 解冻所有参数，使用混合学习率
# CTM: lr=1e-4, Mamba2 SSM: lr=1e-5, MLA: lr=1e-4, MoE: lr=1e-4
optimizer = create_hybrid_optimizer(model, config)

# 训练目标:
# total_loss = task_loss + 0.01 * aux_loss + 0.005 * ctm_tick_loss
```

**验证点**: 联合训练后各模块是否协同工作

### Phase 3: 消融实验 (1-2 天)

对每个数据集运行 A/B/C/D/E 五组实验，记录:

```
Dataset: GSM8K
┌─────────┬──────────┬───────────┬──────────────┐
│  Group  │ Accuracy │ Avg Ticks │ Acc/Efficiency│
├─────────┼──────────┼───────────┼──────────────┤
│ A (OFF) │  72.3%   │    N/A    │     N/A      │
│ B (T=1) │  68.1%   │    1.0    │    68.1      │
│ C (T=2) │  73.5%   │    2.0    │    36.8      │
│ D (T=4) │  75.2%   │    4.0    │    18.8      │
│ E (Ada) │  74.8%   │    2.3    │    32.5  ← 最优│
└─────────┴──────────┴───────────┴──────────────┘
```

### Phase 4: 压力测试 (1 天)

- **对抗样本**: 故意构造模糊/矛盾的输入，观察 CTM 是否增加 ticks
- **长序列**: 128K token 输入，测试 CTM 是否仍能有效工作
- **批处理**: 混合简单/复杂样本的 batch，测试自适应早退的 batch 内一致性

---

## 4. 预期结果

| 指标 | 预期值 | 判定标准 |
|------|--------|----------|
| Adaptive vs Fixed-4 准确率 | ≥ 95% | PASS if E_acc / D_acc >= 0.95 |
| CTM 计算节省 | ≥ 30% | PASS if E_ticks / 4 <= 0.70 |
| 简单任务 1-tick 准确率 | ≥ 90% of 4-tick | PASS on SimpleQA |
| 复杂任务 tick 分布 | 3-4 tick 占 > 60% | PASS on MATH, BBH |
| 同步矩阵熵 vs 任务难度 | Pearson r > 0.5 | 验证早退信号有效性 |

---

## 5. 失败处理预案

| 失败模式 | 原因 | 解决方案 |
|----------|------|----------|
| CTM 始终选择 1 tick | 早退信号过强 | 降低 entropy_thresholds，增加 tick 惩罚 |
| CTM 始终选择 4 ticks | 早退信号过弱 | 提高 entropy_thresholds，增加早退奖励 |
| CTM 导致准确率下降 | 残差注入干扰主模型 | 降低 gate 初始化值，从 0.01 开始 |
| 同步矩阵熵无区分度 | 神经元不收敛 | 增加 CTM 预训练步数，使用对比学习 |
| 梯度消失/爆炸 | RNN 递归过深 | 梯度裁剪阈值降至 0.5，使用 LayerNorm |

---

## 6. 快速验证脚本

```python
# scripts/validate_ctm.py
"""Quick CTM validation: compare CTM=OFF vs CTM=ON on a few samples."""

import torch
from config import LunaConfig
from modeling_luna import LunaUltimate

def validate_ctm_quick():
    config = LunaConfig()
    model = LunaUltimate(config)
    model.eval()

    test_prompts = [
        ("What is 2+2?", "simple"),
        ("Solve ∫x²dx from 0 to 1", "medium"),
        ("Prove that √2 is irrational", "hard"),
    ]

    for prompt, difficulty in test_prompts:
        input_ids = torch.randint(0, 1000, (1, 64))
        with torch.no_grad():
            # CTM OFF
            logits_off, _ = model(input_ids, use_ctm_adaptive=False)
            # CTM ON (adaptive)
            logits_on, _ = model(input_ids, use_ctm_adaptive=True)

        kl_div = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(logits_off, dim=-1),
            torch.nn.functional.softmax(logits_on, dim=-1),
            reduction='batchmean',
        )
        print(f"[{difficulty:>6}] CTM KL divergence: {kl_div.item():.6f}")
        # Expect: simple < medium < hard (CTM has more impact on hard problems)

if __name__ == "__main__":
    validate_ctm_quick()
```

---

## 7. 发布标准

CTM 模块通过验证的条件：

- [ ] Adaptive 模式在 5/7 数据集上达到 Fixed-4 的 95% 准确率
- [ ] 平均 ticks < 2.8 (节省 > 30%)
- [ ] 同步矩阵熵与任务难度正相关 (r > 0.5)
- [ ] 无梯度爆炸/消失 (spike_count < 10 per 1000 steps)
- [ ] 简单任务 (SimpleQA) 1-tick 比例 > 80%