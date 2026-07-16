# Luna 中文文档索引

- [安全与忠诚](./SAFETY.md)
- [全脑架构 WB-HCA](./BRAIN_ARCHITECTURE.md)
- [双轨部署（Ollama + Native）](./DEPLOYMENT.md)
- 英文原文：`docs/SAFETY.md`、`docs/BRAIN_ARCHITECTURE.md`、`docs/ARCHITECTURE.md`、`docs/COST_MODEL.md`、`docs/SCALE_LADDER.md`、`DEPLOYMENT.md`

## 常用命令（中文日志）

```bash
# 安全 + 价值观套件
python scripts/run_safety_tests.py --strict

# 脑区 / 丘脑路由
python scripts/run_neuroarch.py

# 自进化 CI（合格可 --open-pr；上传默认关）
python evolve_ci.py --n-runs 3 --preset tiny

# LoRA 冒烟 / champion 上传就绪
python scripts/lora_finetune.py --preset tiny --steps 3
python scripts/upload_champion.py --model_path checkpoints/champion --dry-run
```

## 定案摘要

- **Q1B** 纯意义优先解码，**无**自回归 fallback（Q3B：坍塌只标记）。
- **三层硬门**：`charter OR values OR ctm` 拒绝。
- **GitHub** = 代码 PR；**权重** = ModelScope（需 `MODELSCOPE_TOKEN` + `--upload-weights`）。
