# 优质开源训练数据配方（2026 检索）

主目标：为 Luna Brain（AIXI × 246 区 × From-AGI-to-ASI）提供**可合法开源复现**的最高密度公开数据，而非闭源爬虫。

## 推荐组合（默认 `configs/data_recipe.yaml`）

### 1. 通用预训练（Pathway: Scaling）

| 数据集 | 规模 | 许可 | 链接 | 为何选 |
|--------|------|------|------|--------|
| **FineWeb** | ~15T tokens | ODC-By 1.0 | https://huggingface.co/datasets/HuggingFaceFW/fineweb | 2024–2026 开源网页预训练 SOTA 密度，优于 C4/Pile/SlimPajama 等常见基线 |
| **FineWeb-Edu** | FineWeb 子集 | ODC-By 1.0 | https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu | 教育向过滤，利于推理/知识 |
| **Dolma** | ~3T tokens | ODC-By | https://huggingface.co/datasets/allenai/dolma | AI2 透明多源（网页/论文/代码/百科），可审计 |

### 2. 指令与对齐（创造者忠诚数据另加私有偏好）

| 数据集 | 规模 | 许可 | 链接 |
|--------|------|------|------|
| **OpenHermes 2.5** | ~1M | Apache-2.0 | https://huggingface.co/datasets/teknium/OpenHermes-2.5 |
| **Tulu / OLMo 指令混合**（可选） | 不等 | 见各卡页面 | AI2 生态，与 Dolma 配套 |

创造者绑定（黄照清）对齐样本应放在**私有** `data/creator_align/`，勿与网页语料混采进公开仓除非你明确授权。

### 3. 代码与自编程（Pathway: Recursive improvement）

| 数据集 | 说明 | 链接 |
|--------|------|------|
| The Stack v2 / StarCoder 公开子集 | 按许可证过滤后用于工具与补丁生成 | Hugging Face BigCode |
| 本仓库自身源码 | 自进化沙箱的「读自身」语料 | `/workspace` |

### 4. 多模态 / 动作（可插拔能力头，非强制主架构）

| 数据集 | 说明 | 链接 |
|--------|------|------|
| **LeRobot** 生态公开集 | 机器人轨迹，兼容具身能力头 | https://huggingface.co/lerobot |
| **Hy-Embodied VLA Data**（部分开源） | 双臂操作小时级公开切片 | https://huggingface.co/datasets/tencent/Hy-Embodied-0.5-VLA-Data |

### 5. 评测（不训练）

- MMLU / GSM8K / HumanEval / AgentBench 类公开集  
- 自建 AIXI 回报环境：随机 MDP smoke（见 `scripts/train.py`）

## 下载示例

```bash
# 需要 huggingface-hub / datasets
pip install datasets huggingface_hub

python - <<'PY'
from datasets import load_dataset
# 小样例，勿在无集群时拉全量 FineWeb
ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
print(next(iter(ds)))
PY
```

## 许可注意

- FineWeb / Dolma：**ODC-By**，衍生模型需遵守数据集条款与署名。  
- 上传权重到 Hugging Face 时，在模型卡中列出训练数据与许可。  
- **不要**把未授权私有数据或旧 ModelScope 闭源权重混进本开源发布。
