# ModelScope / 摩搭：已清空

本仓库**已删除**全部旧 ModelScope 上传/下载脚本与 `luna-ultimate-550b` 相关代码路径。

开源分发改为：

1. GitHub 本仓库（源码）  
2. Hugging Face：`scripts/export_weights.py` + `scripts/publish_hf.py`（需 `HF_TOKEN`）

## 如何删除你账号下旧的摩搭模型仓

本环境**没有** ModelScope API Token，无法代你远程删仓。请登录 [modelscope.cn](https://www.modelscope.cn/) → 模型 → `huang18928827157/luna-ultimate-550b`（或你的旧命名空间）→ 设置 → 删除模型。

若你之后提供 `MODELSCOPE_API_TOKEN`，可再用官方 CLI 删除；当前默认不再维护摩搭同步。
