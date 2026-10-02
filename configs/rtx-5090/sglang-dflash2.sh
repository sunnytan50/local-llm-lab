#!/bin/bash
set -euo pipefail

TARGET="/opt/llm/models/hot/Qwen3.8-27B-NVFP4-RTX5090"
DRAFTER="/opt/llm/models/hot/Qwen3.8-27B-DFlash2"
VENV="/opt/llm/venvs/sglang-dflash2"
CUDA_ROOT="/opt/llm/cuda-13.0-vllm"
CONTEXT="${1:-32768}"

case "$CONTEXT" in
  32768|65536|114688) ;;
  *) echo "Unsupported DFlash 2 context: $CONTEXT" >&2; exit 2 ;;
esac

if find "$TARGET" "$DRAFTER" -type f -name '*.incomplete' -print -quit | grep -q .; then
  echo "Model download is incomplete" >&2
  exit 3
fi

export CUDA_HOME="$CUDA_ROOT"
export PATH="$CUDA_ROOT/bin:$VENV/bin:/usr/lib/wsl/lib:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export LD_LIBRARY_PATH="$CUDA_ROOT/lib:${LD_LIBRARY_PATH:-}"
export MAX_JOBS=2

exec "$VENV/bin/sglang" serve "$TARGET" \
  --host 127.0.0.1 \
  --port 8002 \
  --served-model-name qwen38-nvfp4 \
  --quantization modelopt \
  --kv-cache-dtype fp8_e4m3 \
  --context-length "$CONTEXT" \
  --max-running-requests 1 \
  --max-total-tokens "$CONTEXT" \
  --max-mamba-cache-size 5 \
  --mem-fraction-static 0.85 \
  --attention-backend triton \
  --sampling-backend pytorch \
  --cuda-graph-max-bs-decode 1 \
  --cuda-graph-backend-prefill disabled \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path "$DRAFTER" \
  --speculative-draft-model-quantization unquant \
  --speculative-num-draft-tokens 8 \
  --speculative-draft-kv-cache-dtype fp8_e4m3
