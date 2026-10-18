#!/usr/bin/env bash
set -euo pipefail

UPSTREAM_REPO=${UPSTREAM_REPO:?set the clean official checkout}
TOOL_REPO=${TOOL_REPO:?set the clean CBF-TTT checkout}
TOOL_COMMIT=${TOOL_COMMIT:?set the exact tool commit}
PYTHON_BIN=${PYTHON_BIN:?set the official training Python}
MODEL=${MODEL:?set the Qwen3-4B model directory}
CORPUS_ROOT=${CORPUS_ROOT:?set the independently audited full corpus}
LADDER_ROOT=${LADDER_ROOT:?set the future Stage-1 ladder root}
LADDER_SESSION=${LADDER_SESSION:?set the Stage-1 ladder watcher tmux session}
OUTPUT_ROOT=${OUTPUT_ROOT:?set a new timing output root}
WATCH_ROOT=${WATCH_ROOT:-${OUTPUT_ROOT}.watch}
TIMING_PLAN=${TIMING_PLAN:?set the frozen Stage-1 timing plan}
EXECUTION_ADDENDUM=${EXECUTION_ADDENDUM:?set the frozen 2x5090 addendum}
POLL_SECONDS=${POLL_SECONDS:-60}
MAX_WAIT_SECONDS=${MAX_WAIT_SECONDS:-604800}
EXPECTED_HOSTNAME=${EXPECTED_HOSTNAME:-user-Super-Server}
UPSTREAM_COMMIT=be2324829b0e91c8fd10a74d4b43714fde6676e1

if [[ "$(hostname)" != "$EXPECTED_HOSTNAME" ]]; then
  echo "the Stage-1 timing watcher is restricted to hku-gpu2" >&2; exit 2
fi
if [[ -e "$OUTPUT_ROOT" || -e "$WATCH_ROOT" ]]; then
  echo "refusing to reuse timing or watcher output" >&2; exit 3
fi
if [[ "$POLL_SECONDS" -lt 10 || "$MAX_WAIT_SECONDS" -lt "$POLL_SECONDS" ]]; then
  echo "invalid watcher interval/budget" >&2; exit 2
fi
test "$(git -C "$UPSTREAM_REPO" rev-parse HEAD)" = "$UPSTREAM_COMMIT"
test -z "$(git -C "$UPSTREAM_REPO" status --short)"
test "$(git -C "$TOOL_REPO" rev-parse HEAD)" = "$TOOL_COMMIT"
test -z "$(git -C "$TOOL_REPO" status --short)"
test -x "$PYTHON_BIN"; test -s "$MODEL/config.json"
test -s "$TIMING_PLAN"; test -s "$EXECUTION_ADDENDUM"

mkdir -p "$WATCH_ROOT"
sha256sum "$0" \
  "$TOOL_REPO/scripts/run_in_place_ttt_2x5090_stage1_timing.sh" \
  "$TOOL_REPO/scripts/audit_in_place_ttt_2x5090_stage1_timing.py" \
  "$TIMING_PLAN" "$EXECUTION_ADDENDUM" "$MODEL/config.json" \
  > "$WATCH_ROOT/frozen_inputs.sha256"

write_state() {
  printf '%s\n' "$1" > "$WATCH_ROOT/state.txt.partial"
  mv "$WATCH_ROOT/state.txt.partial" "$WATCH_ROOT/state.txt"
  date --iso-8601=seconds > "$WATCH_ROOT/state_time.txt.partial"
  mv "$WATCH_ROOT/state_time.txt.partial" "$WATCH_ROOT/state_time.txt"
}

snapshot() {
  local prefix=$1
  date --iso-8601=seconds > "$WATCH_ROOT/${prefix}_time.txt"
  if [[ -f "$LADDER_ROOT/state.txt" ]]; then
    cp "$LADDER_ROOT/state.txt" "$WATCH_ROOT/${prefix}_ladder_state.txt"
  else
    : > "$WATCH_ROOT/${prefix}_ladder_state.txt"
  fi
  tmux list-panes -t "$LADDER_SESSION" -F '#{pane_pid} #{pane_dead} #{pane_current_command}' \
    > "$WATCH_ROOT/${prefix}_ladder_handle.txt" 2>&1 || true
}

started=$(date +%s)
while true; do
  write_state waiting_for_audited_stage1_ladder
  snapshot last
  ladder_state=$(cat "$LADDER_ROOT/state.txt" 2>/dev/null || true)
  case "$ladder_state" in
    completed_stage1_ladder)
      if ! "$PYTHON_BIN" - "$LADDER_ROOT/independent_ladder_audit.json" <<'PY'
import json, pathlib, sys
audit = json.loads(pathlib.Path(sys.argv[1]).read_text())
if audit.get("passed") is not True:
    raise SystemExit(1)
PY
      then
        write_state invalid_stage1_ladder
        exit 4
      fi
      break
      ;;
    failed_*|invalid_*|timed_out_*|upstream_*|eight_k_*)
      write_state "upstream_${ladder_state}"
      exit 5
      ;;
  esac
  if ! tmux has-session -t "$LADDER_SESSION" 2>/dev/null; then
    snapshot missing_handle
    write_state stage1_ladder_watcher_handle_missing
    exit 6
  fi
  now=$(date +%s)
  if [[ $((now - started)) -ge "$MAX_WAIT_SECONDS" ]]; then
    write_state timed_out_waiting_for_stage1_ladder
    exit 7
  fi
  sleep "$POLL_SECONDS"
done

write_state launching_stage1_timing
set +e
env \
  UPSTREAM_REPO="$UPSTREAM_REPO" TOOL_REPO="$TOOL_REPO" \
  TOOL_COMMIT="$TOOL_COMMIT" PYTHON_BIN="$PYTHON_BIN" MODEL="$MODEL" \
  CORPUS_ROOT="$CORPUS_ROOT" LADDER_ROOT="$LADDER_ROOT" \
  OUTPUT_ROOT="$OUTPUT_ROOT" TIMING_PLAN="$TIMING_PLAN" \
  EXECUTION_ADDENDUM="$EXECUTION_ADDENDUM" EXPECTED_HOSTNAME="$EXPECTED_HOSTNAME" \
  bash "$TOOL_REPO/scripts/run_in_place_ttt_2x5090_stage1_timing.sh"
status=$?
set -e
printf '%s\n' "$status" > "$WATCH_ROOT/timing_runner_exit_code.txt"
if [[ "$status" -ne 0 ]]; then
  write_state failed_stage1_timing
  exit 8
fi
if [[ ! -s "$OUTPUT_ROOT/independent_audit.json" ]]; then
  write_state missing_stage1_timing_audit
  exit 9
fi
"$PYTHON_BIN" - "$OUTPUT_ROOT/independent_audit.json" <<'PY'
import json, pathlib, sys
if json.loads(pathlib.Path(sys.argv[1]).read_text()).get("passed") is not True:
    raise SystemExit(1)
PY
write_state completed_stage1_timing
