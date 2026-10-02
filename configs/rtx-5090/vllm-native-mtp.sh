#!/bin/bash
set -euo pipefail

TARGET="/opt/llm/models/hot/Qwen3.8-27B-NVFP4-RTX5090"
VENV="/opt/llm/venvs/vllm-dflash2"
CUDA_ROOT="/opt/llm/cuda-13.0-vllm"
CONTEXT="${1:-32768}"

case "$CONTEXT" in
  32768) ;;
  *) echo "Unsupported native MTP context: $CONTEXT" >&2; exit 2 ;;
esac

if find "$TARGET" -type f -name '*.incomplete' -print -quit | grep -q .; then
  echo "Model download is incomplete" >&2
  exit 3
fi

export CUDA_HOME="$CUDA_ROOT"
export PATH="$CUDA_ROOT/bin:$VENV/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export LD_LIBRARY_PATH="$CUDA_ROOT/lib:${LD_LIBRARY_PATH:-}"
export MAX_JOBS=2

exec "$VENV/bin/vllm" serve "$TARGET" \
  --host 127.0.0.1 \
  --port 8002 \
  --served-model-name qwen38-nvfp4 \
  --quantization modelopt \
  --kv-cache-dtype fp8 \
  --trust-remote-code \
  --max-model-len "$CONTEXT" \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.92 \
  --enable-prefix-caching \
  --per-request-spec-decode-metrics summary \
  --speculative-config '{"method":"mtp","num_speculative_tokens":4}' \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml
