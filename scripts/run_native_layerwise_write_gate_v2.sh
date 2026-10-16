#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}
REPO=${REPO:-$(pwd)}
MODEL=${MODEL:?set MODEL}
V1_ROOT=${V1_ROOT:?set V1_ROOT to the final audited V1 directory}
OUTPUT_ROOT=${OUTPUT_ROOT:-$REPO/experiments/cbf_ttt/native_layerwise_write_gate_v2}
STAGE=${STAGE:-smoke}
NUM_SHARDS=${NUM_SHARDS:-2}
DATA_ROOT=${DATA_ROOT:-$V1_ROOT/data}

if [[ "$STAGE" != smoke && "$STAGE" != pilot ]]; then echo "invalid STAGE" >&2; exit 2; fi
mkdir -p "$OUTPUT_ROOT"
pids=()
for ((shard=0; shard<NUM_SHARDS; shard++)); do
  output=$OUTPUT_ROOT/${STAGE}_shard${shard}.jsonl
  [[ ! -e "$output" ]] || { echo "refusing to overwrite $output" >&2; exit 3; }
  CUDA_VISIBLE_DEVICES=$shard "$PYTHON_BIN" -m tasks.native_layerwise_write_gate_v2 collect \
    --data-root "$DATA_ROOT" --v1-root "$V1_ROOT" --model "$MODEL" \
    --plan "$REPO/NATIVE_LAYERWISE_WRITE_GATE_V2_PLAN.md" --stage "$STAGE" \
    --shard "$shard" --num-shards "$NUM_SHARDS" --output "$output" \
    > "$OUTPUT_ROOT/${STAGE}_shard${shard}.log" 2>&1 &
  pids+=("$!")
done
status=0; for pid in "${pids[@]}"; do wait "$pid" || status=1; done
[[ "$status" == 0 ]] || { echo "one or more shards failed" >&2; exit 4; }
inputs=(); for ((shard=0; shard<NUM_SHARDS; shard++)); do inputs+=("$OUTPUT_ROOT/${STAGE}_shard${shard}.jsonl"); done
"$PYTHON_BIN" -m tasks.native_layerwise_write_gate_v2 summarize \
  --inputs "${inputs[@]}" --output "$OUTPUT_ROOT/${STAGE}_summary.json"
"$PYTHON_BIN" -m scripts.audit_native_layerwise_write_gate_v2 \
  --root "$OUTPUT_ROOT" --v1-root "$V1_ROOT" \
  --plan "$REPO/NATIVE_LAYERWISE_WRITE_GATE_V2_PLAN.md" --stage "$STAGE"
