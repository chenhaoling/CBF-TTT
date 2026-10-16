#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}
REPO=${REPO:-$(pwd)}
MODEL=${MODEL:?set MODEL to the native Qwen3-4B checkpoint}
DATA_ROOT=${DATA_ROOT:?set DATA_ROOT to the independently audited V3 corpus}
CONTROLLER_ROOT=${CONTROLLER_ROOT:?set CONTROLLER_ROOT to the audited stopped V4 controllers}
OUTPUT_ROOT=${OUTPUT_ROOT:-$REPO/experiments/cbf_ttt/native_closed_loop_fast_weight_gate_v5}
NUM_SHARDS=${NUM_SHARDS:-4}
GPU_COUNT=${GPU_COUNT:-2}
STAGE=${STAGE:-smoke}
SMOKE_ROOT=${SMOKE_ROOT:-}

if [[ "$NUM_SHARDS" -lt 1 || "$NUM_SHARDS" -gt 4 || "$GPU_COUNT" -lt 1 || "$GPU_COUNT" -gt 2 ]]; then
  echo "NUM_SHARDS must be 1..4 and GPU_COUNT must be 1..2" >&2; exit 2
fi
if [[ "$STAGE" != "smoke" && "$STAGE" != "development" ]]; then
  echo "STAGE must be smoke or development" >&2; exit 2
fi
if [[ -e "$OUTPUT_ROOT/${STAGE}_summary.json" || -e "$OUTPUT_ROOT/${STAGE}_independent_audit.json" ]]; then
  echo "refusing to overwrite completed V5 $STAGE artifacts" >&2; exit 3
fi
"$PYTHON_BIN" -c \
  'import hashlib,json,sys; s=json.load(open(sys.argv[1])); a=json.load(open(sys.argv[2])); h=hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest(); assert not s["development_released_test"] and not s["test_opened"] and a["passed"] and not a["test_released"] and not a["test_opened"] and a["summary_sha256"] == h' \
  "$CONTROLLER_ROOT/training_summary.json" "$CONTROLLER_ROOT/training_independent_audit.json"
if [[ "$STAGE" == "development" ]]; then
  if [[ -z "$SMOKE_ROOT" ]]; then
    echo "set SMOKE_ROOT to the independently audited V5 Phase-A output" >&2; exit 2
  fi
  "$PYTHON_BIN" -c \
    'import hashlib,json,sys; s=json.load(open(sys.argv[1])); a=json.load(open(sys.argv[2])); h=hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest(); assert s["stage"] == "smoke" and s["passed"] and not s["test_opened"] and a["passed"] and a["smoke_gate_passed"] and not a["test_opened"] and a["summary_sha256"] == h' \
    "$SMOKE_ROOT/smoke_summary.json" "$SMOKE_ROOT/smoke_independent_audit.json"
  PLAN_FILE=$REPO/NATIVE_CLOSED_LOOP_FAST_WEIGHT_GATE_V5_PHASE_B_PLAN.md
else
  PLAN_FILE=$REPO/NATIVE_CLOSED_LOOP_FAST_WEIGHT_GATE_V5_PLAN.md
fi
mkdir -p "$OUTPUT_ROOT"
pids=()
for ((shard=0; shard<NUM_SHARDS; shard++)); do
  labels=$OUTPUT_ROOT/${STAGE}_shard${shard}_labels.jsonl
  decisions=$OUTPUT_ROOT/${STAGE}_shard${shard}_decisions.jsonl
  if [[ -e "$labels" || -e "$decisions" ]]; then
    echo "refusing to overwrite $labels or $decisions" >&2; exit 3
  fi
  device=$((shard % GPU_COUNT))
  CUDA_VISIBLE_DEVICES=$device "$PYTHON_BIN" -m tasks.native_closed_loop_fast_weight_gate_v5 collect \
    --data-root "$DATA_ROOT" --model "$MODEL" --controller-root "$CONTROLLER_ROOT" \
    --plan "$PLAN_FILE" --stage "$STAGE" \
    --shard "$shard" --num-shards "$NUM_SHARDS" \
    --output "$labels" --decisions-output "$decisions" \
    > "$OUTPUT_ROOT/${STAGE}_shard${shard}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if [[ "$status" != 0 ]]; then echo "one or more V5 $STAGE shards failed" >&2; exit 4; fi
labels=(); decisions=()
for ((shard=0; shard<NUM_SHARDS; shard++)); do
  labels+=("$OUTPUT_ROOT/${STAGE}_shard${shard}_labels.jsonl")
  decisions+=("$OUTPUT_ROOT/${STAGE}_shard${shard}_decisions.jsonl")
done
"$PYTHON_BIN" -m tasks.native_closed_loop_fast_weight_gate_v5 summarize \
  --labels "${labels[@]}" --decisions "${decisions[@]}" \
  --output "$OUTPUT_ROOT/${STAGE}_summary.json"
"$PYTHON_BIN" -m scripts.audit_native_closed_loop_fast_weight_gate_v5 \
  --root "$OUTPUT_ROOT" --data-root "$DATA_ROOT" --controller-root "$CONTROLLER_ROOT" \
  --plan "$PLAN_FILE" --num-shards "$NUM_SHARDS" --stage "$STAGE"
