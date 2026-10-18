#!/usr/bin/env bash
set -euo pipefail

UPSTREAM_REPO=${UPSTREAM_REPO:?set the clean official checkout}
PYTHON_BIN=${PYTHON_BIN:?set the frozen training Python}
MODEL=${MODEL:?set the exact Qwen3-4B-Base directory}
MODEL_IDENTITY=${MODEL_IDENTITY:?set the passing Base identity JSON}
LEVEL_A_ROOT=${LEVEL_A_ROOT:?set the corrected Level-A root}
LEVEL_A_SESSION=${LEVEL_A_SESSION:?set the corrected Level-A tmux session}
OUTPUT_ROOT=${OUTPUT_ROOT:?set a new corrected Stage-1 ladder root}
WATCH_ROOT=${WATCH_ROOT:-${OUTPUT_ROOT}.watch}
PARENT_PLAN=${PARENT_PLAN:?set the parent baseline plan}
STAGE1_PLAN=${STAGE1_PLAN:?set the corrected Stage-1 resource plan}
INPUT_ADDENDUM=${INPUT_ADDENDUM:?set the corrected resource-input addendum}
TOOL_REPO=${TOOL_REPO:?set the clean tool checkout}
TOOL_COMMIT=${TOOL_COMMIT:?set the exact tool commit}
POLL_SECONDS=${POLL_SECONDS:-60}
MAX_WAIT_SECONDS=${MAX_WAIT_SECONDS:-604800}
EXPECTED_HOSTNAME=${EXPECTED_HOSTNAME:-user-Super-Server}
RUNNER="$TOOL_REPO/scripts/run_in_place_ttt_qwen3_4b_base_stage1_ladder.sh"
AUDITOR="$TOOL_REPO/scripts/audit_in_place_ttt_qwen3_4b_base_stage1_ladder.py"
UPSTREAM_COMMIT=be2324829b0e91c8fd10a74d4b43714fde6676e1

[[ "$(hostname)" == "$EXPECTED_HOSTNAME" ]] || {
  echo "corrected Stage-1 watcher is restricted to hku-gpu2" >&2; exit 2;
}
[[ ! -e "$OUTPUT_ROOT" && ! -e "$WATCH_ROOT" ]] || {
  echo "refusing to reuse corrected Stage-1 roots" >&2; exit 3;
}
test "$(git -C "$TOOL_REPO" rev-parse HEAD)" = "$TOOL_COMMIT"
test -z "$(git -C "$TOOL_REPO" status --short)"
test "$(git -C "$UPSTREAM_REPO" rev-parse HEAD)" = "$UPSTREAM_COMMIT"
test -z "$(git -C "$UPSTREAM_REPO" status --short)"
for path in "$MODEL/config.json" "$MODEL_IDENTITY" "$PARENT_PLAN" \
  "$STAGE1_PLAN" "$INPUT_ADDENDUM" "$RUNNER" "$AUDITOR"; do test -s "$path"; done

mkdir -p "$WATCH_ROOT"
sha256sum "$0" "$RUNNER" "$AUDITOR" "$MODEL/config.json" \
  "$MODEL_IDENTITY" "$PARENT_PLAN" "$STAGE1_PLAN" "$INPUT_ADDENDUM" \
  > "$WATCH_ROOT/static_inputs.sha256"

write_state() {
  printf '%s\n' "$1" > "$WATCH_ROOT/state.txt.partial"
  mv "$WATCH_ROOT/state.txt.partial" "$WATCH_ROOT/state.txt"
  date --iso-8601=seconds > "$WATCH_ROOT/state_time.txt.partial"
  mv "$WATCH_ROOT/state_time.txt.partial" "$WATCH_ROOT/state_time.txt"
}

started=$(date +%s)
while true; do
  write_state waiting_for_corrected_level_a
  status=$(cat "$LEVEL_A_ROOT/status.txt" 2>/dev/null || true)
  case "$status" in
    COMPLETE)
      resource="$LEVEL_A_ROOT/resource_inputs"
      test -s "$LEVEL_A_ROOT/independent_audit.json"
      test -s "$resource/manifest.json"
      test -s "$resource/independent_audit.json"
      write_state auditing_corrected_level_a
      "$PYTHON_BIN" "$AUDITOR" prerequisite \
        --level-a-root "$LEVEL_A_ROOT" \
        --level-a-audit "$LEVEL_A_ROOT/independent_audit.json" \
        --data-manifest "$resource/manifest.json" \
        --data-audit "$resource/independent_audit.json" \
        --output "$WATCH_ROOT/level_a_prerequisite_audit.json"
      break
      ;;
    FAILED_*) write_state "upstream_${status}"; exit 4 ;;
  esac
  if ! tmux has-session -t "$LEVEL_A_SESSION" 2>/dev/null; then
    sleep 5
    status=$(cat "$LEVEL_A_ROOT/status.txt" 2>/dev/null || true)
    [[ "$status" == COMPLETE ]] || {
      write_state level_a_handle_missing; exit 5;
    }
  fi
  now=$(date +%s)
  [[ $((now - started)) -lt "$MAX_WAIT_SECONDS" ]] || {
    write_state timed_out_waiting_for_corrected_level_a; exit 6;
  }
  sleep "$POLL_SECONDS"
done

write_state launching_corrected_stage1_ladder
set +e
env UPSTREAM_REPO="$UPSTREAM_REPO" PYTHON_BIN="$PYTHON_BIN" MODEL="$MODEL" \
  MODEL_IDENTITY="$MODEL_IDENTITY" LEVEL_A_ROOT="$LEVEL_A_ROOT" \
  RESOURCE_INPUT_ROOT="$LEVEL_A_ROOT/resource_inputs" OUTPUT_ROOT="$OUTPUT_ROOT" \
  PARENT_PLAN="$PARENT_PLAN" STAGE1_PLAN="$STAGE1_PLAN" \
  INPUT_ADDENDUM="$INPUT_ADDENDUM" TOOL_REPO="$TOOL_REPO" \
  TOOL_COMMIT="$TOOL_COMMIT" EXPECTED_HOSTNAME="$EXPECTED_HOSTNAME" \
  bash "$RUNNER"
status=$?
set -e
printf '%s\n' "$status" > "$WATCH_ROOT/runner_exit_code.txt"
[[ "$status" -eq 0 ]] || { write_state failed_corrected_stage1_ladder; exit 7; }
"$PYTHON_BIN" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["passed"]' \
  "$OUTPUT_ROOT/independent_ladder_audit.json"
write_state completed_corrected_stage1_ladder
