#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}
REPO=${REPO:-$(pwd)}
MODEL=${MODEL:?set MODEL to the native Qwen3-4B checkpoint}
TOKENIZER=${TOKENIZER:?set TOKENIZER to Qwen3-4B tokenizer}
SOURCE_EPISODES=${SOURCE_EPISODES:?set SOURCE_EPISODES to audited pilot_episodes.jsonl}
OUTPUT_ROOT=${OUTPUT_ROOT:-$REPO/experiments/cbf_ttt/support_write_gate_v1}
STAGE=${STAGE:-smoke}
NUM_SHARDS=${NUM_SHARDS:-2}

if [[ "$STAGE" != smoke && "$STAGE" != pilot ]]; then
  echo "STAGE must be smoke or pilot" >&2
  exit 2
fi
if [[ "$NUM_SHARDS" -lt 1 || "$NUM_SHARDS" -gt 2 ]]; then
  echo "NUM_SHARDS must be 1 or 2 on hku-gpu2" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"
DATA_ROOT=$OUTPUT_ROOT/data
if [[ ! -f "$DATA_ROOT/manifest.json" ]]; then
  "$PYTHON_BIN" -m tasks.build_native_support_write_gate_v1 \
    --source-episodes "$SOURCE_EPISODES" \
    --tokenizer "$TOKENIZER" \
    --plan "$REPO/SUPPORT_AWARE_WRITE_GATE_V1_PLAN.md" \
    --output "$DATA_ROOT"
fi

pids=()
for ((shard=0; shard<NUM_SHARDS; shard++)); do
  output=$OUTPUT_ROOT/${STAGE}_shard${shard}.jsonl
  if [[ -e "$output" ]]; then
    echo "refusing to overwrite $output" >&2
    exit 3
  fi
  CUDA_VISIBLE_DEVICES=$shard "$PYTHON_BIN" -m tasks.native_support_write_gate_v1 collect \
    --data-root "$DATA_ROOT" --model "$MODEL" --stage "$STAGE" \
    --shard "$shard" --num-shards "$NUM_SHARDS" --output "$output" \
    > "$OUTPUT_ROOT/${STAGE}_shard${shard}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if [[ "$status" != 0 ]]; then
  echo "one or more $STAGE shards failed" >&2
  exit 4
fi

inputs=()
for ((shard=0; shard<NUM_SHARDS; shard++)); do
  inputs+=("$OUTPUT_ROOT/${STAGE}_shard${shard}.jsonl")
done
"$PYTHON_BIN" -m tasks.native_support_write_gate_v1 summarize \
  --inputs "${inputs[@]}" --output "$OUTPUT_ROOT/${STAGE}_summary.json"
"$PYTHON_BIN" -m scripts.audit_native_support_write_gate_v1 \
  --root "$OUTPUT_ROOT" --data-root "$DATA_ROOT" \
  --plan "$REPO/SUPPORT_AWARE_WRITE_GATE_V1_PLAN.md" --stage "$STAGE"
