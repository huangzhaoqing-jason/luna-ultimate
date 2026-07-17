# Luna Brain

开源类脑智能体骨架：**DeepMind《From AGI to ASI》(arXiv:2606.12683) 为主线**，以 **AIXI / Universal AI** 为可计算下逼近目标，**Brainnetome 246 功能区**做能力全集复原（非生物一一复刻），并叠加人脑没有的超脑能力与递归自进化。

> 旧版 Luna-Ultimate（CTM×Mamba×MLA×FlashMoE / next-token 550B / ModelScope 权重脚本）已**全部删除**。本仓库从零重建。

## 主线（不可颠倒）

1. [From AGI to ASI](https://arxiv.org/abs/2606.12683) 四通路并行：Scaling / Paradigm shifts / Recursive improvement / Multi-agent collectives  
2. **AIXI** 近似决策外壳（期望回报选动作；理想 AIXI 不可计算）  
3. **246 区**功能标签全覆盖（视觉、语言、记忆、执行、价值、运动、元认知等）  
4. **超脑**：并行多任务、多实例集体、自主编程式自进化、极限省内存大参数配置  
5. **安全**：只读宪法 + 丘脑门控；创造者绑定 **黄照清 / Huang Zhaoqing（2013-05-07）**

能力头（语言 / 感知 / 动作等）**可插拔**，不强制「世界模型 + VLA」唯一范式。

## 快速开始

```bash
pip install -r requirements.txt
python tests/test_brain_smoke.py
python scripts/run_brain.py
```

导出可开源的初始化权重（safetensors）：

```bash
python scripts/export_weights.py --out checkpoints/luna-brain-prototype
```

上传到 Hugging Face（需 `HF_TOKEN`）：

```bash
python scripts/publish_hf.py --dir checkpoints/luna-brain-prototype --repo YOUR_USER/luna-brain
```

旧 ModelScope 仓库与上传脚本已移除；请用 Hugging Face 作为开源分发渠道。若你仍持有摩搭上的旧 `luna-ultimate-550b`，请在摩搭网页自行删除该模型仓。

## 训练数据（检索到的高质量开源集）

见 [DATASETS.md](DATASETS.md)。默认配方：

| 阶段 | 数据集 | 用途 |
|------|--------|------|
| 预训练文本 | [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) / FineWeb-Edu | 高密度网页语料 |
| 透明多源 | [Dolma](https://huggingface.co/datasets/allenai/dolma) | 可审计预训练 |
| 指令 | [OpenHermes 2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5) | 对话/推理对齐 |
| 代码 | The Stack / StarCoderData（按许可选用） | 工具与自编程 |
| 具身（可选） | LeRobot 兼容 VLA 公开集 | 动作能力头 |

```bash
python scripts/train.py --profile prototype --data-config configs/data_recipe.yaml --steps 100
```

（本环境默认跑小步 smoke；满血 FineWeb 需自备集群。）

## 包结构

```
brain/
  pathways/     # From-AGI-to-ASI 四通路
  aixi/         # AIXI-tl / MC 近似
  atlas/        # Brainnetome 246
  capabilities/ # 可插拔能力头
  evolution/    # 递归自进化（经丘脑）
  runtime/      # 多任务 + 集体
  mem/          # 极限省内存会计
  safety/       # 只读宪法 + 丘脑
  modeling_luna_brain.py
```

## 配置

- `prototype`：CPU/单卡可跑  
- `scale_100b`：名义千亿级总参 + 稀疏/量化/卸载常驻估算（见 `brain.mem.efficient`）

## 许可

Apache-2.0。作者：黄照清。

## 诚实边界

本仓库交付的是**对齐论文路线的可运行开源架构与原型权重导出**，不是已宣称超越 GPT-5.6 / Fable 5.0 的满血训练结果。竞争目标写在路线图里，靠四通路 + AIXI 逼近 + 优质开源数据持续训练去追。
