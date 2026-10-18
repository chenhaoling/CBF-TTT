#!/usr/bin/env bash
set -euo pipefail

UPSTREAM_REPO=${UPSTREAM_REPO:?set the clean official checkout}
PYTHON_BIN=${PYTHON_BIN:?set the baseline training Python}
MODEL=${MODEL:?set the Qwen3-4B model directory}
EIGHT_K_ROOT=${EIGHT_K_ROOT:?set the existing 8K watcher output root}
EIGHT_K_SESSION=${EIGHT_K_SESSION:?set the existing 8K watcher tmux session}
RESOURCE_INPUT_ROOT=${RESOURCE_INPUT_ROOT:?set the audited exact input root}
OUTPUT_ROOT=${OUTPUT_ROOT:?set a new Stage-1 ladder output root}
WATCH_ROOT=${WATCH_ROOT:-${OUTPUT_ROOT}.watch}
PLAN=${PLAN:?set the parent baseline plan}
ADDENDUM=${ADDENDUM:?set the frozen 2x5090 execution addendum}
INPUT_ADDENDUM=${INPUT_ADDENDUM:?set the exact-input addendum}
TOOL_REPO=${TOOL_REPO:?set the clean CBF-TTT checkout}
TOOL_COMMIT=${TOOL_COMMIT:?set the exact tool commit}
POLL_SECONDS=${POLL_SECONDS:-60}
MAX_WAIT_SECONDS=${MAX_WAIT_SECONDS:-604800}
EXPECTED_HOSTNAME=${EXPECTED_HOSTNAME:-user-Super-Server}
UPSTREAM_COMMIT=be2324829b0e91c8fd10a74d4b43714fde6676e1

if [[ "$(hostname)" != "$EXPECTED_HOSTNAME" ]]; then
  echo "the Stage-1 ladder watcher is restricted to hku-gpu2" >&2
  exit 2
fi
if [[ -e "$OUTPUT_ROOT" || -e "$WATCH_ROOT" ]]; then
  echo "refusing to reuse the ladder or watcher output root" >&2
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
test -s "$MODEL/config.json"; test -s "$PLAN"; test -s "$ADDENDUM"
test -s "$INPUT_ADDENDUM"
test -s "$RESOURCE_INPUT_ROOT/manifest.json"
test -s "$RESOURCE_INPUT_ROOT/independent_audit.json"
test -d "$EIGHT_K_ROOT"

mkdir -p "$WATCH_ROOT"
sha256sum \
  "$0" "$TOOL_REPO/scripts/run_in_place_ttt_2x5090_stage1_ladder.sh" \
  "$TOOL_REPO/scripts/audit_in_place_ttt_2x5090_stage1_ladder.py" \
  "$PLAN" "$ADDENDUM" "$INPUT_ADDENDUM" \
  "$RESOURCE_INPUT_ROOT/manifest.json" \
  "$RESOURCE_INPUT_ROOT/independent_audit.json" "$MODEL/config.json" \
  > "$WATCH_ROOT/frozen_inputs.sha256"

write_state() {
  local value=$1
  printf '%s\n' "$value" > "$WATCH_ROOT/state.txt.partial"
  mv "$WATCH_ROOT/state.txt.partial" "$WATCH_ROOT/state.txt"
  date --iso-8601=seconds > "$WATCH_ROOT/state_time.txt.partial"
  mv "$WATCH_ROOT/state_time.txt.partial" "$WATCH_ROOT/state_time.txt"
}

snapshot() {
  local prefix=$1
  date --iso-8601=seconds > "$WATCH_ROOT/${prefix}_time.txt"
  if [[ -f "$EIGHT_K_ROOT/state.txt" ]]; then
    cp "$EIGHT_K_ROOT/state.txt" "$WATCH_ROOT/${prefix}_eight_k_state.txt"
  else
    : > "$WATCH_ROOT/${prefix}_eight_k_state.txt"
  fi
  tmux list-panes -t "$EIGHT_K_SESSION" -F '#{pane_pid} #{pane_dead} #{pane_current_command}' \
    > "$WATCH_ROOT/${prefix}_eight_k_handle.txt" 2>&1 || true
}

started=$(date +%s)
while true; do
  write_state waiting_for_audited_8k
  snapshot last
  eight_k_state=$(cat "$EIGHT_K_ROOT/state.txt" 2>/dev/null || true)
  case "$eight_k_state" in
    completed_8k_both_arms)
      write_state auditing_8k_prerequisite
      if ! "$PYTHON_BIN" \
        "$TOOL_REPO/scripts/audit_in_place_ttt_2x5090_stage1_ladder.py" \
        prerequisite --eight-k-root "$EIGHT_K_ROOT" \
        --output "$WATCH_ROOT/eight_k_prerequisite_audit.json"; then
        write_state invalid_8k_prerequisite
        exit 4
      fi
      break
      ;;
    failed_*|invalid_*|timed_out_*)
      write_state "upstream_${eight_k_state}"
      exit 5
      ;;
  esac
  if ! tmux has-session -t "$EIGHT_K_SESSION" 2>/dev/null; then
    snapshot missing_handle
    write_state eight_k_watcher_handle_missing
    exit 6
  fi
  now=$(date +%s)
  if [[ $((now - started)) -ge "$MAX_WAIT_SECONDS" ]]; then
    write_state timed_out_waiting_for_audited_8k
    exit 7
  fi
  sleep "$POLL_SECONDS"
done

write_state launching_stage1_ladder
set +e
env \
  UPSTREAM_REPO="$UPSTREAM_REPO" PYTHON_BIN="$PYTHON_BIN" MODEL="$MODEL" \
  EIGHT_K_ROOT="$EIGHT_K_ROOT" RESOURCE_INPUT_ROOT="$RESOURCE_INPUT_ROOT" \
  OUTPUT_ROOT="$OUTPUT_ROOT" PLAN="$PLAN" ADDENDUM="$ADDENDUM" \
  INPUT_ADDENDUM="$INPUT_ADDENDUM" TOOL_REPO="$TOOL_REPO" \
  TOOL_COMMIT="$TOOL_COMMIT" EXPECTED_HOSTNAME="$EXPECTED_HOSTNAME" \
  bash "$TOOL_REPO/scripts/run_in_place_ttt_2x5090_stage1_ladder.sh"
status=$?
set -e
printf '%s\n' "$status" > "$WATCH_ROOT/ladder_runner_exit_code.txt"
if [[ "$status" -ne 0 ]]; then
  write_state failed_stage1_ladder
  exit 8
fi
if [[ "$(cat "$OUTPUT_ROOT/state.txt" 2>/dev/null || true)" \
      != completed_stage1_ladder ]]; then
  write_state invalid_stage1_ladder_state
  exit 9
fi
"$PYTHON_BIN" - "$OUTPUT_ROOT/independent_ladder_audit.json" <<'PY'
import json, pathlib, sys
audit = json.loads(pathlib.Path(sys.argv[1]).read_text())
if audit.get("passed") is not True:
    raise SystemExit(1)
PY
write_state completed_stage1_ladder
