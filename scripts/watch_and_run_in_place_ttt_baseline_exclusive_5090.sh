#!/usr/bin/env bash
set -euo pipefail

UPSTREAM_REPO=${UPSTREAM_REPO:?set the clean official checkout}
PYTHON_BIN=${PYTHON_BIN:?set the baseline training Python}
MODEL=${MODEL:?set the Qwen3-4B model directory}
DATA=${DATA:?set the packed pilot JSONL}
OUTPUT_ROOT=${OUTPUT_ROOT:?set a new watcher output root}
PLAN=${PLAN:?set the frozen baseline plan path}
ADDENDUM=${ADDENDUM:?set the frozen exclusive retry addendum path}
TOOL_REPO=${TOOL_REPO:?set the CBF-TTT tool checkout}
POLL_SECONDS=${POLL_SECONDS:-60}
MAX_WAIT_SECONDS=${MAX_WAIT_SECONDS:-86400}
MAX_IDLE_MEMORY_MIB=${MAX_IDLE_MEMORY_MIB:-512}
UPSTREAM_COMMIT=be2324829b0e91c8fd10a74d4b43714fde6676e1

if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "refusing to reuse $OUTPUT_ROOT" >&2
  exit 3
fi
mkdir -p "$OUTPUT_ROOT"
sha256sum "$0" > "$OUTPUT_ROOT/watcher_script.sha256"
sha256sum "$PLAN" "$ADDENDUM" > "$OUTPUT_ROOT/frozen_inputs.sha256"

write_state() {
  local value=$1
  printf '%s\n' "$value" > "$OUTPUT_ROOT/state.txt.tmp"
  mv "$OUTPUT_ROOT/state.txt.tmp" "$OUTPUT_ROOT/state.txt"
}

snapshot() {
  local prefix=$1
  date --iso-8601=seconds > "$OUTPUT_ROOT/${prefix}_time.txt"
  nvidia-smi \
    --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits > "$OUTPUT_ROOT/${prefix}_gpu.csv"
  nvidia-smi \
    --query-compute-apps=gpu_uuid,pid,used_memory,process_name \
    --format=csv,noheader,nounits > "$OUTPUT_ROOT/${prefix}_processes.csv"
}

eligible() {
  local prefix=$1
  snapshot "$prefix"
  "$PYTHON_BIN" - "$OUTPUT_ROOT/${prefix}_gpu.csv" \
    "$OUTPUT_ROOT/${prefix}_processes.csv" "$MAX_IDLE_MEMORY_MIB" <<'PY'
import csv, pathlib, sys
gpu_path, process_path, limit = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), int(sys.argv[3])
rows = list(csv.reader(gpu_path.open()))
if len(rows) != 2:
    raise SystemExit(1)
for expected, row in enumerate(rows):
    if int(row[0].strip()) != expected or row[1].strip() != "NVIDIA GeForce RTX 5090":
        raise SystemExit(1)
    if int(row[3].strip()) > limit:
        raise SystemExit(1)
if process_path.read_text().strip():
    raise SystemExit(1)
PY
}

wait_for_exclusive() {
  local arm=$1
  local started now elapsed
  started=$(date +%s)
  while true; do
    write_state "waiting_${arm}"
    if eligible "last_${arm}"; then
      sleep 5
      if eligible "preflight_${arm}"; then
        test "$(git -C "$UPSTREAM_REPO" rev-parse HEAD)" = "$UPSTREAM_COMMIT"
        test -z "$(git -C "$UPSTREAM_REPO" status --short)"
        return 0
      fi
    fi
    now=$(date +%s)
    elapsed=$((now - started))
    if [[ "$elapsed" -ge "$MAX_WAIT_SECONDS" ]]; then
      write_state "timed_out_waiting_${arm}"
      return 1
    fi
    sleep "$POLL_SECONDS"
  done
}

run_arm() {
  local arm=$1
  local output=$2
  write_state "running_${arm}"
  set +e
  env \
    UPSTREAM_REPO="$UPSTREAM_REPO" \
    PYTHON_BIN="$PYTHON_BIN" \
    MODEL="$MODEL" \
    DATA="$DATA" \
    OUTPUT_ROOT="$output" \
    ARM="$arm" \
    RESOURCE_PROFILE=native_expandable_segments \
    SEQ_LEN=8192 \
    GLOBAL_BATCH_SIZE=2 \
    PLAN="$PLAN" \
    ADDENDUM="$ADDENDUM" \
    TOOL_REPO="$TOOL_REPO" \
    bash "$TOOL_REPO/scripts/run_in_place_ttt_baseline_resource_pilot.sh"
  local status=$?
  set -e
  printf '%s\n' "$status" > "$OUTPUT_ROOT/${arm}_runner_exit_code.txt"
  return "$status"
}

validate_success() {
  local summary=$1
  "$PYTHON_BIN" - "$summary" <<'PY'
import json, math, pathlib, sys
payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
if not payload.get("completed") or payload.get("exit_code") != 0:
    raise SystemExit(1)
for key in ("last_loss", "last_gradient_norm"):
    value = payload.get(key)
    if value is None or not math.isfinite(float(value)):
        raise SystemExit(1)
PY
}

if ! wait_for_exclusive no_ttt; then
  exit 4
fi
if ! run_arm no_ttt "$OUTPUT_ROOT/resource_8k_no_ttt"; then
  write_state failed_no_ttt
  exit 5
fi
if ! validate_success "$OUTPUT_ROOT/resource_8k_no_ttt/summary.json"; then
  write_state invalid_no_ttt
  exit 6
fi

if ! wait_for_exclusive ttt; then
  exit 7
fi
if ! run_arm ttt "$OUTPUT_ROOT/resource_8k_ttt"; then
  write_state failed_ttt
  exit 8
fi
if ! validate_success "$OUTPUT_ROOT/resource_8k_ttt/summary.json"; then
  write_state invalid_ttt
  exit 9
fi

write_state completed_both_arms
