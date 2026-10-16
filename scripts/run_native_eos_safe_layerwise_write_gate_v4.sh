#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}
REPO=${REPO:-$(pwd)}
MODEL=${MODEL:?set MODEL to the native Qwen3-4B checkpoint}
DATA_ROOT=${DATA_ROOT:?set DATA_ROOT to the independently audited 624-group corpus}
V3_ROOT=${V3_ROOT:?set V3_ROOT to the stopped and audited V3 confirmation}
OUTPUT_ROOT=${OUTPUT_ROOT:-$REPO/experiments/cbf_ttt/native_eos_safe_layerwise_write_gate_v4}
STAGE=${STAGE:-train}
NUM_SHARDS=${NUM_SHARDS:-2}
GPU_COUNT=${GPU_COUNT:-2}

if [[ "$STAGE" != train && "$STAGE" != development ]]; then
  echo "STAGE must be train or development" >&2; exit 2
fi
if [[ "$NUM_SHARDS" -lt 1 || "$NUM_SHARDS" -gt 4 || "$GPU_COUNT" -lt 1 || "$GPU_COUNT" -gt 2 ]]; then
  echo "NUM_SHARDS must be 1..4 and GPU_COUNT must be 1..2" >&2; exit 2
fi
"$PYTHON_BIN" -c \
  'import json,sys; a=json.load(open(sys.argv[1])); d=json.load(open(sys.argv[2])); c=d["accuracy_and_eos_safe_oracle"]; assert a["passed"] and not a["test_opened"] and d["v3_gate_remains_failed"] and c["gain_group_bootstrap_95ci"][0] > 0 and c["eos_rate_delta"] <= 0' \
  "$V3_ROOT/confirmation_independent_audit.json" "$V3_ROOT/confirmation_failure_diagnostic.json"
mkdir -p "$OUTPUT_ROOT"
pids=()
for ((shard=0; shard<NUM_SHARDS; shard++)); do
  labels=$OUTPUT_ROOT/${STAGE}_shard${shard}_labels.jsonl
  features=$OUTPUT_ROOT/${STAGE}_shard${shard}_features.jsonl
  if [[ -e "$labels" || -e "$features" ]]; then
    echo "refusing to overwrite $labels or $features" >&2; exit 3
  fi
  device=$((shard % GPU_COUNT))
  CUDA_VISIBLE_DEVICES=$device "$PYTHON_BIN" -m tasks.native_eos_safe_layerwise_write_gate_v4 collect \
    --data-root "$DATA_ROOT" --model "$MODEL" --stage "$STAGE" \
    --plan "$REPO/NATIVE_EOS_SAFE_LAYERWISE_WRITE_GATE_V4_PLAN.md" --v3-root "$V3_ROOT" \
    --shard "$shard" --num-shards "$NUM_SHARDS" \
    --output "$labels" --features-output "$features" \
    > "$OUTPUT_ROOT/${STAGE}_shard${shard}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if [[ "$status" != 0 ]]; then echo "one or more V4 shards failed" >&2; exit 4; fi
inputs=()
for ((shard=0; shard<NUM_SHARDS; shard++)); do inputs+=("$OUTPUT_ROOT/${STAGE}_shard${shard}_labels.jsonl"); done
"$PYTHON_BIN" -m tasks.native_eos_safe_layerwise_write_gate_v4 summarize \
  --inputs "${inputs[@]}" --output "$OUTPUT_ROOT/${STAGE}_summary.json"
"$PYTHON_BIN" -m scripts.audit_native_eos_safe_layerwise_write_gate_v4 \
  --root "$OUTPUT_ROOT" --data-root "$DATA_ROOT" \
  --plan "$REPO/NATIVE_EOS_SAFE_LAYERWISE_WRITE_GATE_V4_PLAN.md" \
  --v3-root "$V3_ROOT" --stage "$STAGE"
