#!/usr/bin/env bash
set -euo pipefail

UPSTREAM_REPO=${UPSTREAM_REPO:?set the clean official checkout}
PYTHON_BIN=${PYTHON_BIN:?set the baseline training Python}
MODEL=${MODEL:?set the Qwen3-4B model directory}
DATA=${DATA:?set the packed pilot JSONL}
OUTPUT_ROOT=${OUTPUT_ROOT:?set a new retry output root}
PLAN=${PLAN:?set the frozen baseline plan path}
ADDENDUM=${ADDENDUM:?set the frozen asymmetric-idle addendum path}
TOOL_REPO=${TOOL_REPO:?set the CBF-TTT tool checkout}
UPSTREAM_COMMIT=be2324829b0e91c8fd10a74d4b43714fde6676e1

if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "refusing to reuse $OUTPUT_ROOT" >&2
  exit 3
fi
mkdir -p "$OUTPUT_ROOT"
sha256sum "$0" > "$OUTPUT_ROOT/retry_script.sha256"
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
    --query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits > "$OUTPUT_ROOT/${prefix}_gpu.csv"
  nvidia-smi \
    --query-compute-apps=gpu_uuid,pid,used_memory,process_name \
    --format=csv,noheader,nounits > "$OUTPUT_ROOT/${prefix}_processes.csv"
}

eligible() {
  local prefix=$1
  snapshot "$prefix"
  "$PYTHON_BIN" - "$OUTPUT_ROOT/${prefix}_gpu.csv" \
    "$OUTPUT_ROOT/${prefix}_processes.csv" <<'PY'
import csv
import pathlib
import sys

gpu_path, process_path = map(pathlib.Path, sys.argv[1:])
gpus = list(csv.reader(gpu_path.open()))
processes = list(csv.reader(process_path.open()))
if len(gpus) != 2:
    raise SystemExit("expected exactly two GPUs")

by_uuid = {}
for expected_index, row in enumerate(gpus):
    index = int(row[0].strip())
    uuid = row[1].strip()
    name = row[2].strip()
    used_mib = int(row[4].strip())
    if index != expected_index or name != "NVIDIA GeForce RTX 5090":
        raise SystemExit("unexpected GPU inventory")
    limit = 768 if index == 0 else 128
    if used_mib > limit:
        raise SystemExit(f"GPU{index} uses {used_mib} MiB, above {limit} MiB")
    by_uuid[uuid] = index

per_gpu = {0: [], 1: []}
for row in processes:
    if not row or not row[0].strip():
        continue
    uuid = row[0].strip()
    if uuid not in by_uuid:
        raise SystemExit(f"process reported for unknown GPU {uuid}")
    per_gpu[by_uuid[uuid]].append(row)

if len(per_gpu[0]) > 1:
    raise SystemExit("GPU0 has more than one unrelated compute process")
if per_gpu[0] and int(per_gpu[0][0][2].strip()) > 672:
    raise SystemExit("GPU0 unrelated process exceeds 672 MiB")
if per_gpu[1]:
    raise SystemExit("GPU1 has an unrelated compute process")
PY

  if pgrep -af 'tasks/train_torch.py|torch.distributed.run' \
      > "$OUTPUT_ROOT/${prefix}_baseline_processes.txt"; then
    echo "an earlier baseline training process is alive" >&2
    return 1
  fi
  test "$(git -C "$UPSTREAM_REPO" rev-parse HEAD)" = "$UPSTREAM_COMMIT"
  test -z "$(git -C "$UPSTREAM_REPO" status --short)"
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
import json
import math
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
if not payload.get("completed") or payload.get("exit_code") != 0:
    raise SystemExit(1)
for key in ("last_loss", "last_gradient_norm"):
    value = payload.get(key)
    if value is None or not math.isfinite(float(value)):
        raise SystemExit(1)
PY
}

write_state preflight_no_ttt
if ! eligible preflight_no_ttt; then
  write_state ineligible_no_ttt
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

write_state preflight_ttt
if ! eligible preflight_ttt; then
  write_state ineligible_ttt
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
