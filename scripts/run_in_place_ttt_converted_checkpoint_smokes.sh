#!/usr/bin/env bash
set -euo pipefail

UPSTREAM_REPO=${UPSTREAM_REPO:?set the clean official checkout}
EVAL_PYTHON=${EVAL_PYTHON:?set the isolated OpenCompass Python}
TOKENIZER=${TOKENIZER:?set the original Qwen3-4B tokenizer directory}
OUTPUT_ROOT=${OUTPUT_ROOT:?set the completed baseline resource root}
PLAN=${PLAN:?set the frozen baseline plan}
TOOL_REPO=${TOOL_REPO:?set the CBF-TTT tool checkout}
TOKENIZER_AUDIT_ADDENDUM=${TOKENIZER_AUDIT_ADDENDUM:-}
SMOKE_SUFFIX=${SMOKE_SUFFIX:-smoke}
REQUIRE_ZERO_TOKENIZER_REGEX_WARNINGS=${REQUIRE_ZERO_TOKENIZER_REGEX_WARNINGS:-0}
UPSTREAM_COMMIT=be2324829b0e91c8fd10a74d4b43714fde6676e1

if [[ ! "$SMOKE_SUFFIX" =~ ^[a-z0-9_]+$ ]]; then
  echo "invalid SMOKE_SUFFIX: $SMOKE_SUFFIX" >&2; exit 3
fi

test "$(git -C "$UPSTREAM_REPO" rev-parse HEAD)" = "$UPSTREAM_COMMIT"
test -z "$(git -C "$UPSTREAM_REPO" status --short)"
test "$(cat "$OUTPUT_ROOT/conversion_state.txt")" = completed_both_arms

state_path="$OUTPUT_ROOT/converted_${SMOKE_SUFFIX}_state.txt"
printf '%s\n' preflight > "$state_path"
date --iso-8601=seconds > "$OUTPUT_ROOT/converted_${SMOKE_SUFFIX}_preflight_time.txt"
nvidia-smi \
  --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader,nounits \
  > "$OUTPUT_ROOT/converted_${SMOKE_SUFFIX}_preflight_gpu.csv"
nvidia-smi \
  --query-compute-apps=gpu_uuid,pid,used_memory,process_name \
  --format=csv,noheader,nounits \
  > "$OUTPUT_ROOT/converted_${SMOKE_SUFFIX}_preflight_processes.csv"

"$EVAL_PYTHON" - "$OUTPUT_ROOT/converted_${SMOKE_SUFFIX}_preflight_gpu.csv" \
  "$OUTPUT_ROOT/converted_${SMOKE_SUFFIX}_preflight_processes.csv" <<'PY'
import csv
import pathlib
import sys

gpus = list(csv.reader(pathlib.Path(sys.argv[1]).open()))
if len(gpus) != 2:
    raise SystemExit("expected exactly two GPUs")
for expected, row in enumerate(gpus):
    if int(row[0].strip()) != expected:
        raise SystemExit("unexpected GPU index")
    if row[1].strip() != "NVIDIA GeForce RTX 5090":
        raise SystemExit("unexpected GPU type")
if int(gpus[0][3].strip()) > 512:
    raise SystemExit("GPU0 is not idle enough for checkpoint smoke")
if pathlib.Path(sys.argv[2]).read_text().strip():
    raise SystemExit("a compute process is already present")
PY

validate_summary() {
  local path=$1
  "$EVAL_PYTHON" - "$path" "$REQUIRE_ZERO_TOKENIZER_REGEX_WARNINGS" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
require_zero_warnings = bool(int(sys.argv[2]))
if not payload.get("completed"):
    raise SystemExit("RULER smoke did not complete")
if payload.get("task_count") != 13:
    raise SystemExit("unexpected RULER task count")
if not payload.get("macro_matches_official"):
    raise SystemExit("official and independently recomputed scores disagree")
if not payload.get("dependencies", {}).get("model_hash_manifest_sha256"):
    raise SystemExit("converted checkpoint hash manifest is not bound")
if not payload.get("dependencies", {}).get("tokenizer_json_sha256"):
    raise SystemExit("evaluation tokenizer hash is not bound")
if require_zero_warnings and payload.get("tokenizer_regex_warning_count") != 0:
    raise SystemExit("tokenizer regex warning remains")
PY
}

for arm in no_ttt ttt; do
  arm_root="$OUTPUT_ROOT/resource_8k_${arm}"
  model="$arm_root/hf_ckpt"
  hash_manifest="$arm_root/hf_files.sha256"
  eval_root="$OUTPUT_ROOT/ruler_4k_${arm}_step1_${SMOKE_SUFFIX}"
  test -s "$model/config.json"
  test -s "$hash_manifest"
  test ! -e "$eval_root"
  printf 'running_%s\n' "$arm" > "$state_path"
  env \
    UPSTREAM_REPO="$UPSTREAM_REPO" \
    PYTHON_BIN="$EVAL_PYTHON" \
    MODEL="$model" \
    TOKENIZER="$TOKENIZER" \
    MODEL_NAME="qwen3_4b_${arm}_step1_${SMOKE_SUFFIX}" \
    MODEL_HASH_MANIFEST="$hash_manifest" \
    TOKENIZER_AUDIT_ADDENDUM="$TOKENIZER_AUDIT_ADDENDUM" \
    OUTPUT_ROOT="$eval_root" \
    PLAN="$PLAN" \
    TOOL_REPO="$TOOL_REPO" \
    bash "$TOOL_REPO/scripts/run_in_place_ttt_ruler_smoke.sh"
  "$EVAL_PYTHON" "$TOOL_REPO/scripts/summarize_in_place_ttt_ruler_smoke.py" \
    --root "$eval_root" --output "$eval_root/summary.json"
  validate_summary "$eval_root/summary.json"
done

date --iso-8601=seconds > "$OUTPUT_ROOT/converted_${SMOKE_SUFFIX}_completed_at.txt"
printf '%s\n' completed_both_arms > "$state_path"
