#!/usr/bin/env bash
set -euo pipefail

UPSTREAM_REPO=${UPSTREAM_REPO:?set the clean official checkout}
PYTHON_BIN=${PYTHON_BIN:?set the baseline training Python}
MODEL=${MODEL:?set the Qwen3-4B model directory}
DATA=${DATA:?set the exact 8K packed pilot JSONL}
OUTPUT_ROOT=${OUTPUT_ROOT:?set a new watcher output root}
PLAN=${PLAN:?set the parent baseline plan}
ADDENDUM=${ADDENDUM:?set the frozen 2x5090 addendum}
TOOL_REPO=${TOOL_REPO:?set the clean CBF-TTT tool checkout}
TOOL_COMMIT=${TOOL_COMMIT:?set the exact tool commit}
POLL_SECONDS=${POLL_SECONDS:-60}
MAX_WAIT_SECONDS=${MAX_WAIT_SECONDS:-259200}
EXPECTED_HOSTNAME=${EXPECTED_HOSTNAME:-user-Super-Server}
UPSTREAM_COMMIT=be2324829b0e91c8fd10a74d4b43714fde6676e1

if [[ "$(hostname)" != "$EXPECTED_HOSTNAME" ]]; then
  echo "this watcher is restricted to hku-gpu2" >&2
  exit 2
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "refusing to reuse $OUTPUT_ROOT" >&2
  exit 3
fi
if [[ "$POLL_SECONDS" -lt 10 || "$MAX_WAIT_SECONDS" -lt "$POLL_SECONDS" ]]; then
  echo "invalid watcher interval/budget" >&2
  exit 2
fi
test "$(git -C "$UPSTREAM_REPO" rev-parse HEAD)" = "$UPSTREAM_COMMIT"
test -z "$(git -C "$UPSTREAM_REPO" status --short)"
test "$(git -C "$TOOL_REPO" rev-parse HEAD)" = "$TOOL_COMMIT"
test -z "$(git -C "$TOOL_REPO" status --short)"
test -s "$MODEL/config.json"; test -s "$DATA"; test -s "$PLAN"; test -s "$ADDENDUM"

mkdir -p "$OUTPUT_ROOT"
sha256sum "$0" "$PLAN" "$ADDENDUM" "$DATA" "$MODEL/config.json" \
  > "$OUTPUT_ROOT/frozen_inputs.sha256"
"$PYTHON_BIN" "$TOOL_REPO/scripts/audit_in_place_ttt_2x5090_configs.py" \
  --output "$OUTPUT_ROOT/config_audit.json" > "$OUTPUT_ROOT/config_audit.stdout"

write_state() {
  local value=$1
  printf '%s\n' "$value" > "$OUTPUT_ROOT/state.txt.partial"
  mv "$OUTPUT_ROOT/state.txt.partial" "$OUTPUT_ROOT/state.txt"
  date --iso-8601=seconds > "$OUTPUT_ROOT/state_time.txt.partial"
  mv "$OUTPUT_ROOT/state_time.txt.partial" "$OUTPUT_ROOT/state_time.txt"
}

snapshot() {
  local prefix=$1
  date --iso-8601=seconds > "$OUTPUT_ROOT/${prefix}_time.txt"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
    --format=csv,noheader,nounits > "$OUTPUT_ROOT/${prefix}_gpu.csv"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader,nounits > "$OUTPUT_ROOT/${prefix}_processes.csv"
  # /proc/meminfo is mode 0444.  `cp` propagates that mode when it creates the
  # destination, so the next polling iteration cannot overwrite its own file.
  # Shell redirection applies the watcher umask and keeps repeated snapshots
  # writable by the watcher process.
  cat /proc/meminfo > "$OUTPUT_ROOT/${prefix}_host.txt"
  "$PYTHON_BIN" - "$OUTPUT_ROOT/${prefix}_conflicts.txt" <<'PY'
import pathlib, shlex, sys
targets = {"build_in_place_ttt_public_corpus.py", "audit_in_place_ttt_public_corpus.py"}
rows = []
for process in pathlib.Path("/proc").glob("[0-9]*"):
    try:
        argv = [item.decode(errors="replace") for item in
                (process / "cmdline").read_bytes().split(b"\0") if item]
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    if any(pathlib.Path(argument).name in targets for argument in argv):
        rows.append((int(process.name), argv))
pathlib.Path(sys.argv[1]).write_text("".join(
    f"{pid} {shlex.join(argv)}\n" for pid, argv in sorted(rows)
))
PY
}

eligible() {
  local prefix=$1
  snapshot "$prefix"
  if ! "$PYTHON_BIN" - "$OUTPUT_ROOT" "$prefix" <<'PY'
import csv, pathlib, re, sys
root, prefix = pathlib.Path(sys.argv[1]), sys.argv[2]
with (root / f"{prefix}_gpu.csv").open(newline="") as handle:
    rows = [row for row in csv.reader(handle) if row]
if len(rows) != 2:
    raise SystemExit(1)
for expected, row in enumerate(rows):
    index, name, total, used, _ = [value.strip() for value in row]
    if (int(index) != expected or "RTX 5090" not in name
            or float(total) < 32_000 or float(used) > 128):
        raise SystemExit(1)
if (root / f"{prefix}_processes.csv").read_text().strip():
    raise SystemExit(1)
if (root / f"{prefix}_conflicts.txt").read_text().strip():
    raise SystemExit(1)
host = (root / f"{prefix}_host.txt").read_text()
match = re.search(r"^MemAvailable:\s+(\d+)\s+kB$", host, re.MULTILINE)
if match is None or int(match.group(1)) < 48 * 1024 * 1024:
    raise SystemExit(1)
PY
  then
    return 1
  fi
  [[ "$(git -C "$UPSTREAM_REPO" rev-parse HEAD)" == "$UPSTREAM_COMMIT" ]] \
    || return 1
  [[ -z "$(git -C "$UPSTREAM_REPO" status --short)" ]] || return 1
  [[ "$(git -C "$TOOL_REPO" rev-parse HEAD)" == "$TOOL_COMMIT" ]] || return 1
  [[ -z "$(git -C "$TOOL_REPO" status --short)" ]] || return 1
  return 0
}

wait_for_resources() {
  local arm=$1
  local started now
  started=$(date +%s)
  while true; do
    write_state "waiting_8k_${arm}"
    if eligible "last_${arm}"; then
      sleep 5
      if eligible "preflight_${arm}"; then
        return 0
      fi
    fi
    now=$(date +%s)
    if [[ $((now - started)) -ge "$MAX_WAIT_SECONDS" ]]; then
      write_state "timed_out_waiting_8k_${arm}"
      return 1
    fi
    sleep "$POLL_SECONDS"
  done
}

run_arm() {
  local arm=$1
  local config output status
  config="$TOOL_REPO/configs/baseline_reproduction/qwen3_4b_2x5090_100m_stage1_${arm}.yaml"
  output="$OUTPUT_ROOT/resource_8k_${arm}"
  write_state "running_8k_${arm}"
  set +e
  env \
    UPSTREAM_REPO="$UPSTREAM_REPO" PYTHON_BIN="$PYTHON_BIN" \
    MODEL="$MODEL" DATA="$DATA" OUTPUT_ROOT="$output" ARM="$arm" \
    RESOURCE_PROFILE=activation_offload SEQ_LEN=8192 GLOBAL_BATCH_SIZE=2 \
    PLAN="$PLAN" ADDENDUM="$ADDENDUM" CONFIG_OVERRIDE="$config" \
    TOOL_REPO="$TOOL_REPO" EXPECTED_HOSTNAME="$EXPECTED_HOSTNAME" \
    bash "$TOOL_REPO/scripts/run_in_place_ttt_baseline_resource_pilot.sh"
  status=$?
  set -e
  printf '%s\n' "$status" > "$OUTPUT_ROOT/${arm}_runner_exit_code.txt"
  return "$status"
}

validate_arm() {
  local arm=$1
  "$PYTHON_BIN" - "$OUTPUT_ROOT/resource_8k_${arm}" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
summary = json.loads((root / "summary.json").read_text())
audit = json.loads((root / "independent_audit.json").read_text())
if (not summary.get("completed")
        or summary.get("activation_offload_gate_passed") is not True
        or audit.get("passed") is not True):
    raise SystemExit(1)
PY
}

for arm in no_ttt ttt; do
  if ! wait_for_resources "$arm"; then
    exit 4
  fi
  if ! run_arm "$arm"; then
    write_state "failed_8k_${arm}"
    exit 5
  fi
  if ! validate_arm "$arm"; then
    write_state "invalid_8k_${arm}"
    exit 6
  fi
done

write_state completed_8k_both_arms
