#!/usr/bin/env bash
set -euo pipefail

SFT_CODE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${SFT_DATA:?Set SFT_DATA to the prepared all-folds dataset directory}"
SFT_MODEL="${SFT_MODEL:-/kaggle/input/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1}"
SFT_OUT="${SFT_OUT:-/kaggle/working/sft-run}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

# Measured on the 175 GiB host. Use mmap when RAM cannot retain the 95 GiB table.
case "${SFT_PLE_MODE:-resident}" in
  resident) SFT_PLE_ARGS=(--ple-resident --ple-cache-gib 0) ;;
  mmap) SFT_PLE_ARGS=(--ple-cache-gib 1) ;;
  *) echo 'SFT_PLE_MODE must be resident or mmap' >&2; exit 2 ;;
esac

exec python -u "$SFT_CODE_ROOT/run.py" \
  --mode fit --model "$SFT_MODEL" \
  --keep "$SFT_CODE_ROOT/artifacts/keep-256-fold0.json" \
  --data "$SFT_DATA" --profile "${SFT_PROFILE:-rtx-pro-6000-96gb}" \
  --out "$SFT_OUT" --session-minutes "${SFT_MINUTES:-900}" \
  --checkpoint-group "${SFT_CHECKPOINT_GROUP:-3}" \
  --ple-block 8192 --no-ple-checkpoint --gated-norm-block 262144 \
  --rms-block-mib 40 --gdn-chunk-tokens "${SFT_GDN_CHUNK_TOKENS:-8192}" \
  --gdn-block-tokens "${SFT_GDN_BLOCK_TOKENS:-0}" \
  --attention-projection-block "${SFT_ATTENTION_PROJECTION_BLOCK:-32768}" \
  "${SFT_PLE_ARGS[@]}" "$@"
