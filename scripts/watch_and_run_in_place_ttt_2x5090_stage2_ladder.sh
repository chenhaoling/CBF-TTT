#!/usr/bin/env bash
set -euo pipefail

UPSTREAM_REPO=${UPSTREAM_REPO:?set the clean official checkout}
TOOL_REPO=${TOOL_REPO:?set the clean CBF-TTT checkout}
TOOL_COMMIT=${TOOL_COMMIT:?set the exact tool commit}
PYTHON_BIN=${PYTHON_BIN:?set the baseline training Python}
STAGE1_ROOT=${STAGE1_ROOT:?set the future paired formal Stage-1 root}
STAGE1_SESSION=${STAGE1_SESSION:?set the formal Stage-1 watcher tmux session}
RESOURCE_INPUT_ROOT=${RESOURCE_INPUT_ROOT:?set the audited exact input root}
OUTPUT_ROOT=${OUTPUT_ROOT:?set a new Stage-2 ladder output root}
WATCH_ROOT=${WATCH_ROOT:-${OUTPUT_ROOT}.watch}
BASELINE_PLAN=${BASELINE_PLAN:?set the parent baseline plan}
RESOURCE_PLAN=${RESOURCE_PLAN:?set the frozen Stage-2 resource plan}
EXECUTION_ADDENDUM=${EXECUTION_ADDENDUM:?set the frozen 2x5090 addendum}
INPUT_ADDENDUM=${INPUT_ADDENDUM:?set the exact-input addendum}
POLL_SECONDS=${POLL_SECONDS:-60}
MAX_WAIT_SECONDS=${MAX_WAIT_SECONDS:-604800}
EXPECTED_HOSTNAME=${EXPECTED_HOSTNAME:-user-Super-Server}
UPSTREAM_COMMIT=be2324829b0e91c8fd10a74d4b43714fde6676e1

if [[ "$(hostname)" != "$EXPECTED_HOSTNAME" ]]; then
  echo "the Stage-2 ladder watcher is restricted to hku-gpu2" >&2; exit 2
fi
if [[ -e "$OUTPUT_ROOT" || -e "$WATCH_ROOT" ]]; then
  echo "refusing to reuse Stage-2 ladder or watcher output" >&2; exit 3
fi
if [[ "$POLL_SECONDS" -lt 10 || "$MAX_WAIT_SECONDS" -lt "$POLL_SECONDS" ]]; then
  echo "invalid watcher interval/budget" >&2; exit 2
fi
test "$(git -C "$UPSTREAM_REPO" rev-parse HEAD)" = "$UPSTREAM_COMMIT"
test -z "$(git -C "$UPSTREAM_REPO" status --short)"
test "$(git -C "$TOOL_REPO" rev-parse HEAD)" = "$TOOL_COMMIT"
test -z "$(git -C "$TOOL_REPO" status --short)"
test -x "$PYTHON_BIN"
test -s "$RESOURCE_INPUT_ROOT/manifest.json"
test -s "$RESOURCE_INPUT_ROOT/independent_audit.json"
test -s "$BASELINE_PLAN"; test -s "$RESOURCE_PLAN"
test -s "$EXECUTION_ADDENDUM"; test -s "$INPUT_ADDENDUM"

mkdir -p "$WATCH_ROOT"
sha256sum \
  "$0" "$TOOL_REPO/scripts/run_in_place_ttt_2x5090_stage2_ladder.sh" \
  "$TOOL_REPO/scripts/audit_in_place_ttt_2x5090_stage2_ladder.py" \
  "$BASELINE_PLAN" "$RESOURCE_PLAN" "$EXECUTION_ADDENDUM" \
  "$INPUT_ADDENDUM" "$RESOURCE_INPUT_ROOT/manifest.json" \
  "$RESOURCE_INPUT_ROOT/independent_audit.json" \
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
  for source in "$STAGE1_ROOT/state.txt" "${STAGE1_ROOT}.watch/state.txt"; do
    name=$(basename "$(dirname "$source")")_$(basename "$source")
    if [[ -f "$source" ]]; then
      cp "$source" "$WATCH_ROOT/${prefix}_${name}"
    else
      : > "$WATCH_ROOT/${prefix}_${name}"
    fi
  done
  tmux list-panes -t "$STAGE1_SESSION" \
    -F '#{pane_pid} #{pane_dead} #{pane_current_command}' \
    > "$WATCH_ROOT/${prefix}_stage1_handle.txt" 2>&1 || true
}

stage1_valid() {
  "$PYTHON_BIN" - "$STAGE1_ROOT/independent_audit.json" <<'PY'
import json, pathlib, sys
audit = json.loads(pathlib.Path(sys.argv[1]).read_text())
if (audit.get("protocol") != "in_place_ttt_2x5090_100m_stage1_audit_v1"
        or audit.get("passed") is not True or audit.get("errors") != []
        or set(audit.get("arms", {})) != {"no_ttt", "ttt"}):
    raise SystemExit(1)
PY
}

started=$(date +%s)
while true; do
  write_state waiting_for_audited_formal_stage1
  snapshot last
  stage1_state=$(cat "$STAGE1_ROOT/state.txt" 2>/dev/null || true)
  if [[ "$stage1_state" == completed_100m_stage1 ]]; then
    if ! stage1_valid; then
      write_state invalid_formal_stage1
      exit 4
    fi
    break
  fi
  upstream_state=$(cat "${STAGE1_ROOT}.watch/state.txt" 2>/dev/null || true)
  case "$upstream_state" in
    failed_*|invalid_*|timed_out_*|upstream_*|stage1_*_missing)
      write_state "upstream_${upstream_state}"
      exit 5
      ;;
  esac
  if ! tmux has-session -t "$STAGE1_SESSION" 2>/dev/null; then
    snapshot missing_handle
    write_state formal_stage1_watcher_handle_missing
    exit 6
  fi
  now=$(date +%s)
  if [[ $((now - started)) -ge "$MAX_WAIT_SECONDS" ]]; then
    write_state timed_out_waiting_for_formal_stage1
    exit 7
  fi
  sleep "$POLL_SECONDS"
done

write_state launching_stage2_ladder
set +e
env \
  UPSTREAM_REPO="$UPSTREAM_REPO" PYTHON_BIN="$PYTHON_BIN" \
  STAGE1_ROOT="$STAGE1_ROOT" RESOURCE_INPUT_ROOT="$RESOURCE_INPUT_ROOT" \
  OUTPUT_ROOT="$OUTPUT_ROOT" BASELINE_PLAN="$BASELINE_PLAN" \
  RESOURCE_PLAN="$RESOURCE_PLAN" EXECUTION_ADDENDUM="$EXECUTION_ADDENDUM" \
  INPUT_ADDENDUM="$INPUT_ADDENDUM" TOOL_REPO="$TOOL_REPO" \
  TOOL_COMMIT="$TOOL_COMMIT" EXPECTED_HOSTNAME="$EXPECTED_HOSTNAME" \
  bash "$TOOL_REPO/scripts/run_in_place_ttt_2x5090_stage2_ladder.sh"
status=$?
set -e
printf '%s\n' "$status" > "$WATCH_ROOT/stage2_ladder_runner_exit_code.txt"
if [[ "$status" -ne 0 ]]; then
  write_state failed_stage2_ladder
  exit 8
fi
if [[ "$(cat "$OUTPUT_ROOT/state.txt" 2>/dev/null || true)" \
      != completed_stage2_ladder \
      || ! -s "$OUTPUT_ROOT/independent_ladder_audit.json" ]]; then
  write_state invalid_stage2_ladder_output
  exit 9
fi
"$PYTHON_BIN" - "$OUTPUT_ROOT/independent_ladder_audit.json" <<'PY'
import json, pathlib, sys
audit = json.loads(pathlib.Path(sys.argv[1]).read_text())
if (audit.get("protocol") != "in_place_ttt_2x5090_stage2_ladder_audit_v1"
        or audit.get("passed") is not True or audit.get("errors") != []
        or set(audit.get("boundaries", {})) != {
            "65536_no_ttt", "65536_ttt", "131072_no_ttt", "131072_ttt",
        }):
    raise SystemExit(1)
PY
write_state completed_stage2_ladder
