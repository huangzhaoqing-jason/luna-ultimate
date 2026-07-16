# 优化阶梯：JEPA 总控 → 1b 真训 → 评测 → 效率

诚实边界：当前默认仍是 `tiny` 原型，**不等于** GPT-5.x。目标是先追平开源 7B 指令模型。

## 1. JEPA 总控（默认开）

- `config.jepa_control_enabled=True`，`route_mode="jepa"`
- `JEPAController` → `thalamus.plan_from_jepa` → CTM ticks / MoE budget / 脑区开关
- Meaning / Motor / World-Action 条件于 `jepa_ctrl`
- 对照：`route_mode="heuristic"` 回到关键词丘脑

冒烟：

```bash
python scripts/smoke_jepa_control.py
```

## 2. 1b 短训 + 蒸馏钩子

真实文本（本地文件或目录 `.txt`）：

```bash
# 准备语料（一行一条）
mkdir -p data && printf 'hello world\ngradient descent\n' > data/corpus.txt

# 1b 短训入口
python train.py --preset 1b --modality text --data_path data/corpus.txt \
  --max_steps 100 --seq_len 128 --output_dir checkpoints/1b_short

# 或经 scale_train
python scripts/scale_train.py --preset 1b --max_steps 100 \
  --data_path data/corpus.txt --distill
```

训后蒸馏（Ollama 学生）：

```bash
python train.py --preset tiny --smoke --distill --distill_steps 3
# 等价于调用 export/distill_ollama.py
```

无 `--data_path` 时退回随机 DummyDataset——能力上不去。

## 3. 评测门禁

```bash
python scripts/run_eval_proxy.py --preset tiny --output eval_proxy.json
```

报告含：`quality_proxy`、`collapse_rate`、`jepa_uncertainty_mean`、`peak_vram_gb`、`$/MTok`。  
evolve 晋升应对齐：质量↑ 且 `safety` 套件 = 1.0。

## 4. 效率与 serve 成本日志

`serve/luna_server.py` 每次生成打印：

```
[luna_serve][cost] peak_vram_gb=... $/MTok=... jepa_unc=... ticks=...
```

响应 JSON 含 `cost`、`jepa_uncertainty`、`ctm_ticks_plan`。

## 5. ModelScope 权重（不上 GitHub）

**默认行为**：`train.py` 训练成功收尾后自动跑门禁并上传（仓库名与 GitHub 对齐：`luna-ultimate`）。

门禁（`scripts/post_train_gate_and_upload.py`）全部通过才上传：
1. checkpoint 可加载 + 权重无 NaN/Inf  
2. 短 forward logits 有限  
3. safety 套件全绿  

```bash
# 训练（默认自动上传；需 MODELSCOPE_TOKEN）
python train.py --preset tiny --smoke --data_path data/corpus.txt \
  --output_dir checkpoints/scale/tiny

# 仅本地、不上传
python train.py --preset tiny --smoke --no_auto_upload

# 手动重跑门禁+上传
python scripts/post_train_gate_and_upload.py \
  --checkpoint checkpoints/scale/tiny/smoke_final.pt
```

无 token → `ready_no_token`（不假装已上传）。命名空间默认 `huang18928827157`。
