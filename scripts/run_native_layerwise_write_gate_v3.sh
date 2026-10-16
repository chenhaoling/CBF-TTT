#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}
REPO=${REPO:-$(pwd)}
MODEL=${MODEL:?set MODEL to the native Qwen3-4B checkpoint}
DATA_ROOT=${DATA_ROOT:?set DATA_ROOT to the independently audited V3 corpus}
OUTPUT_ROOT=${OUTPUT_ROOT:-$REPO/experiments/cbf_ttt/native_layerwise_write_gate_v3}
STAGE=${STAGE:-smoke}
NUM_SHARDS=${NUM_SHARDS:-2}

if [[ "$STAGE" != smoke && "$STAGE" != confirmation ]]; then
  echo "STAGE must be smoke or confirmation" >&2; exit 2
fi
if [[ "$NUM_SHARDS" -lt 1 || "$NUM_SHARDS" -gt 2 ]]; then
  echo "NUM_SHARDS must be 1 or 2" >&2; exit 2
fi
mkdir -p "$OUTPUT_ROOT"
pids=()
for ((shard=0; shard<NUM_SHARDS; shard++)); do
  labels=$OUTPUT_ROOT/${STAGE}_shard${shard}_labels.jsonl
  features=$OUTPUT_ROOT/${STAGE}_shard${shard}_features.jsonl
  if [[ -e "$labels" || -e "$features" ]]; then
    echo "refusing to overwrite $labels or $features" >&2; exit 3
  fi
  CUDA_VISIBLE_DEVICES=$shard "$PYTHON_BIN" -m tasks.native_layerwise_write_gate_v3 collect \
    --data-root "$DATA_ROOT" --model "$MODEL" --stage "$STAGE" \
    --shard "$shard" --num-shards "$NUM_SHARDS" \
    --output "$labels" --features-output "$features" \
    > "$OUTPUT_ROOT/${STAGE}_shard${shard}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if [[ "$status" != 0 ]]; then echo "one or more V3 shards failed" >&2; exit 4; fi
inputs=()
for ((shard=0; shard<NUM_SHARDS; shard++)); do inputs+=("$OUTPUT_ROOT/${STAGE}_shard${shard}_labels.jsonl"); done
"$PYTHON_BIN" -m tasks.native_layerwise_write_gate_v3 summarize \
  --inputs "${inputs[@]}" --output "$OUTPUT_ROOT/${STAGE}_summary.json"
"$PYTHON_BIN" -m scripts.audit_native_layerwise_write_gate_v3 \
  --root "$OUTPUT_ROOT" --data-root "$DATA_ROOT" \
  --plan "$REPO/NATIVE_LAYERWISE_WRITE_GATE_V3_PLAN.md" --stage "$STAGE"
