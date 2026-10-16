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
CONTROLLER_ROOT=${CONTROLLER_ROOT:-}

if [[ "$STAGE" != train && "$STAGE" != development && "$STAGE" != sealed_test ]]; then
  echo "STAGE must be train, development, or sealed_test" >&2; exit 2
fi
if [[ "$NUM_SHARDS" -lt 1 || "$NUM_SHARDS" -gt 4 || "$GPU_COUNT" -lt 1 || "$GPU_COUNT" -gt 2 ]]; then
  echo "NUM_SHARDS must be 1..4 and GPU_COUNT must be 1..2" >&2; exit 2
fi
"$PYTHON_BIN" -c \
  'import json,sys; a=json.load(open(sys.argv[1])); d=json.load(open(sys.argv[2])); c=d["accuracy_and_eos_safe_oracle"]; assert a["passed"] and not a["test_opened"] and d["v3_gate_remains_failed"] and c["gain_group_bootstrap_95ci"][0] > 0 and c["eos_rate_delta"] <= 0' \
  "$V3_ROOT/confirmation_independent_audit.json" "$V3_ROOT/confirmation_failure_diagnostic.json"
if [[ "$STAGE" == development || "$STAGE" == sealed_test ]]; then
  "$PYTHON_BIN" -c \
    'import hashlib,json,sys; s=json.load(open(sys.argv[1])); a=json.load(open(sys.argv[2])); h=hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest(); assert s["passed"] and a["passed"] and not a["test_opened"] and a["summary_sha256"] == h' \
    "$OUTPUT_ROOT/train_summary.json" "$OUTPUT_ROOT/train_independent_audit.json"
fi
if [[ "$STAGE" == sealed_test ]]; then
  if [[ -z "$CONTROLLER_ROOT" ]]; then echo "sealed_test requires CONTROLLER_ROOT" >&2; exit 2; fi
  if compgen -G "$OUTPUT_ROOT/sealed_test_*" > /dev/null; then
    echo "sealed_test artifacts already exist; refusing a second scoring run" >&2; exit 3
  fi
  "$PYTHON_BIN" -c \
    'import hashlib,json,sys; s=json.load(open(sys.argv[1])); a=json.load(open(sys.argv[2])); h=hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest(); assert s["development_released_test"] and not s["test_opened"] and a["passed"] and a["test_released"] and not a["test_opened"] and a["summary_sha256"] == h' \
    "$CONTROLLER_ROOT/training_summary.json" "$CONTROLLER_ROOT/training_independent_audit.json"
  "$PYTHON_BIN" -c \
    'import hashlib,json,os,pathlib,sys; c=pathlib.Path(sys.argv[1]); o=str(pathlib.Path(sys.argv[2]).resolve()); s=c/"training_summary.json"; a=c/"training_independent_audit.json"; p=c/"sealed_test_release.json"; value={"artifact_root":o,"controller_summary_sha256":hashlib.sha256(s.read_bytes()).hexdigest(),"controller_independent_audit_sha256":hashlib.sha256(a.read_bytes()).hexdigest()}; fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o444); os.write(fd,(json.dumps(value,sort_keys=True)+"\n").encode()); os.close(fd)' \
    "$CONTROLLER_ROOT" "$OUTPUT_ROOT"
fi
mkdir -p "$OUTPUT_ROOT"
pids=()
for ((shard=0; shard<NUM_SHARDS; shard++)); do
  labels=$OUTPUT_ROOT/${STAGE}_shard${shard}_labels.jsonl
  features=$OUTPUT_ROOT/${STAGE}_shard${shard}_features.jsonl
  if [[ -e "$labels" || -e "$features" ]]; then
    echo "refusing to overwrite $labels or $features" >&2; exit 3
  fi
  device=$((shard % GPU_COUNT))
  controller_args=()
  if [[ -n "$CONTROLLER_ROOT" ]]; then controller_args=(--controller-root "$CONTROLLER_ROOT"); fi
  CUDA_VISIBLE_DEVICES=$device "$PYTHON_BIN" -m tasks.native_eos_safe_layerwise_write_gate_v4 collect \
    --data-root "$DATA_ROOT" --model "$MODEL" --stage "$STAGE" \
    --plan "$REPO/NATIVE_EOS_SAFE_LAYERWISE_WRITE_GATE_V4_PLAN.md" --v3-root "$V3_ROOT" \
    "${controller_args[@]}" \
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
  --v3-root "$V3_ROOT" --stage "$STAGE" "${controller_args[@]}"
