# ModelScope release mirror

- Model: https://modelscope.cn/models/huang18928827157/luna-ultimate
- GitHub: https://github.com/huangzhaoqing-jason/luna-ultimate
- Sync: `python scripts/sync_modelscope.py`

`config.json` / `model.safetensors.index.json` / `MANIFEST.json` are committed.
Weight shards (`*.safetensors` / `*.bin`) are downloaded on demand and gitignored.

Current published preset: **tiny** (research smoke). Full **550b** training uses the same code path with `--preset 550b` and a ModelScope dataset.
