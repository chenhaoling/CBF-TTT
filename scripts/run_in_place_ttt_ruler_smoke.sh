#!/usr/bin/env bash
set -euo pipefail

UPSTREAM_REPO=${UPSTREAM_REPO:?set the clean official checkout}
PYTHON_BIN=${PYTHON_BIN:?set the isolated OpenCompass Python}
MODEL=${MODEL:?set the unmodified Qwen3-4B model directory}
TOKENIZER=${TOKENIZER:-$MODEL}
MODEL_NAME=${MODEL_NAME:-qwen3_4b_base}
MODEL_HASH_MANIFEST=${MODEL_HASH_MANIFEST:-}
TOKENIZER_POLICY=${TOKENIZER_POLICY:-qwen_original_tokenizer_mistral_fix_false_v1}
TOKENIZER_AUDIT_ADDENDUM=${TOKENIZER_AUDIT_ADDENDUM:-}
OUTPUT_ROOT=${OUTPUT_ROOT:?set a new output root}
PLAN=${PLAN:?set the frozen baseline plan}
TOOL_REPO=${TOOL_REPO:?set the CBF-TTT tool checkout}
UPSTREAM_COMMIT=be2324829b0e91c8fd10a74d4b43714fde6676e1
CONFIG=$TOOL_REPO/eval_config/baseline_reproduction/qwen3_4b_base_4k_smoke.py

if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "refusing to overwrite $OUTPUT_ROOT" >&2; exit 3
fi
test "$(git -C "$UPSTREAM_REPO" rev-parse HEAD)" = "$UPSTREAM_COMMIT"
test -z "$(git -C "$UPSTREAM_REPO" status --short)"
test -s "$MODEL/config.json"; test -s "$MODEL/tokenizer.json"
test -s "$TOKENIZER/tokenizer.json"; test -s "$TOKENIZER/tokenizer_config.json"
test -s "$CONFIG"; test -s "$PLAN"
if [[ -n "$MODEL_HASH_MANIFEST" ]]; then
  test -s "$MODEL_HASH_MANIFEST"
fi
if [[ -n "$TOKENIZER_AUDIT_ADDENDUM" ]]; then
  test -s "$TOKENIZER_AUDIT_ADDENDUM"
fi
mkdir -p "$OUTPUT_ROOT/opencompass"

"$PYTHON_BIN" - "$UPSTREAM_REPO" "$MODEL" "$TOKENIZER" "$CONFIG" "$PLAN" \
  "$OUTPUT_ROOT/run_metadata.json" "$MODEL_NAME" "$MODEL_HASH_MANIFEST" \
  "$TOKENIZER_POLICY" "$TOKENIZER_AUDIT_ADDENDUM" <<'PY'
import hashlib, json, pathlib, subprocess, sys
(
    upstream,
    model,
    tokenizer,
    config,
    plan,
    output,
    model_name,
    model_hash_manifest,
    tokenizer_policy,
    tokenizer_audit_addendum,
) = sys.argv[1:]
sha = lambda path: hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
model_tokenizer_sha = sha(pathlib.Path(model) / "tokenizer.json")
tokenizer_sha = sha(pathlib.Path(tokenizer) / "tokenizer.json")
if model_tokenizer_sha != tokenizer_sha:
    raise SystemExit("model and evaluation tokenizer.json hashes differ")
metadata = {
    "protocol": "in_place_ttt_ruler_4k_smoke_v2",
    "upstream_commit": subprocess.check_output(
        ["git", "-C", upstream, "rev-parse", "HEAD"], text=True).strip(),
    "upstream_clean": not subprocess.check_output(
        ["git", "-C", upstream, "status", "--short"], text=True).strip(),
    "model": model,
    "model_name": model_name,
    "tokenizer": tokenizer,
    "tokenizer_policy": tokenizer_policy,
    "fix_mistral_regex": False,
    "num_samples_per_task": 1,
    "context_length": 4096,
    "plan_sha256": sha(plan),
    "config_sha256": sha(config),
    "model_config_sha256": sha(pathlib.Path(model) / "config.json"),
    "model_tokenizer_json_sha256": model_tokenizer_sha,
    "tokenizer_json_sha256": tokenizer_sha,
    "tokenizer_config_sha256": sha(
        pathlib.Path(tokenizer) / "tokenizer_config.json"
    ),
    "tokenizer_audit_addendum_sha256": (
        sha(tokenizer_audit_addendum) if tokenizer_audit_addendum else None
    ),
    "model_hash_manifest_sha256": (
        sha(model_hash_manifest) if model_hash_manifest else None
    ),
}
pathlib.Path(output).write_text(json.dumps(metadata, indent=2) + "\n")
PY

printf 'timestamp,index,memory_used_mib,memory_total_mib,utilization_gpu\n' > "$OUTPUT_ROOT/gpu.csv"
(
  while [[ ! -e "$OUTPUT_ROOT/monitor.stop" ]]; do
    timestamp=$(date +%s.%N)
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits | while IFS=',' read -r index used total util; do
        printf '%s,%s,%s,%s,%s\n' "$timestamp" "${index// /}" "${used// /}" \
          "${total// /}" "${util// /}" >> "$OUTPUT_ROOT/gpu.csv"
      done
    sleep 0.5
  done
) &
monitor_pid=$!
cleanup() {
  touch "$OUTPUT_ROOT/monitor.stop"
  wait "$monitor_pid" 2>/dev/null || true
}
trap cleanup EXIT

set +e
(
  cd "$UPSTREAM_REPO"
  export PATH="$(dirname "$PYTHON_BIN"):$PATH"
  export PYTHONPATH="$UPSTREAM_REPO:$TOOL_REPO${PYTHONPATH:+:$PYTHONPATH}"
  export CUDA_VISIBLE_DEVICES=0
  export BASELINE_EVAL_MODEL="$MODEL"
  export BASELINE_EVAL_TOKENIZER="$TOKENIZER"
  export BASELINE_EVAL_NAME="$MODEL_NAME"
  export BASELINE_EVAL_NUM_SAMPLES=1
  export TOKENIZERS_PARALLELISM=false
  /usr/bin/time -v "$PYTHON_BIN" "$TOOL_REPO/tasks/run_opencompass.py" \
    "$CONFIG" --work-dir "$OUTPUT_ROOT/opencompass" --debug
) > "$OUTPUT_ROOT/eval.log" 2>&1
status=$?
set -e
printf '%s\n' "$status" > "$OUTPUT_ROOT/exit_code.txt"
cleanup
trap - EXIT
exit "$status"
