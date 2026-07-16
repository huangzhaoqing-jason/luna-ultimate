#!/usr/bin/env bash
# 发布蒸馏版到本地 Ollama（默认不 push）
# 用法：
#   bash scripts/publish_ollama.sh export/out/luna-distill
#   PUSH=1 bash scripts/publish_ollama.sh export/out/luna-distill  # 需 ollama login

set -euo pipefail
DIR="${1:-export/out/luna-distill}"
NAME="${OLLAMA_MODEL_NAME:-luna}"

if ! command -v ollama >/dev/null 2>&1; then
  echo "[publish_ollama] ollama 未安装。请先安装 https://ollama.com 后重试。"
  echo "[publish_ollama] 蒸馏权重仍可用: $DIR"
  exit 0
fi

if [[ ! -f "$DIR/Modelfile" ]]; then
  echo "[publish_ollama] 缺少 $DIR/Modelfile — 先跑 python3 export/distill_ollama.py"
  exit 2
fi

# 若尚无 GGUF，提示转换
if ! ls "$DIR"/*.gguf >/dev/null 2>&1; then
  echo "[publish_ollama] 未找到 GGUF。请按 $DIR/CONVERT.md 转换后再 create。"
  echo "[publish_ollama] 也可先用 native: python3 scripts/run_serve.py"
  exit 0
fi

echo "[publish_ollama] ollama create $NAME -f $DIR/Modelfile"
( cd "$DIR" && ollama create "$NAME" -f Modelfile )

if [[ "${PUSH:-0}" == "1" ]]; then
  echo "[publish_ollama] ollama push $NAME"
  ollama push "$NAME"
else
  echo "[publish_ollama] 跳过 push（设 PUSH=1 开启）。本地: ollama run $NAME"
fi
