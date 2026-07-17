# ModelScope / 摩搭

## 当前开源模型仓（已上传）

- **https://modelscope.cn/models/huang18928827157/luna-brain**
- 内容：Luna Brain 原型权重（`model.safetensors` + `config.json` + `configuration.json` + README）
- 源码：本 GitHub 仓库（Apache-2.0）

上传命令：

```bash
export MODELSCOPE_API_TOKEN=...
python scripts/export_weights.py
python scripts/publish_modelscope.py \
  --dir checkpoints/luna-brain-prototype \
  --repo huang18928827157/luna-brain
```

## 旧仓删除

API Token **不能**删除模型仓（摩搭限制：须网页删除）。

请登录 https://www.modelscope.cn → 模型 `huang18928827157/luna-ultimate` → 设置 → 删除。

新架构权重请使用 **`luna-brain`**，勿再依赖旧 `luna-ultimate` 权重脚本（已从源码移除）。
