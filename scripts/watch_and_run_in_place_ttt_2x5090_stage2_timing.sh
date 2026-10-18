#!/usr/bin/env bash
set -euo pipefail

UPSTREAM_REPO=${UPSTREAM_REPO:?set the clean official checkout}
TOOL_REPO=${TOOL_REPO:?set the clean CBF-TTT checkout}
TOOL_COMMIT=${TOOL_COMMIT:?set the exact tool commit}
PYTHON_BIN=${PYTHON_BIN:?set the official training Python}
CORPUS_ROOT=${CORPUS_ROOT:?set the independently audited full corpus}
STAGE1_ROOT=${STAGE1_ROOT:?set the completed paired formal Stage-1 root}
STAGE2_LADDER_ROOT=${STAGE2_LADDER_ROOT:?set the future Stage-2 ladder root}
STAGE2_LADDER_SESSION=${STAGE2_LADDER_SESSION:?set the Stage-2 ladder watcher tmux session}
OUTPUT_ROOT=${OUTPUT_ROOT:?set a new Stage-2 timing output root}
WATCH_ROOT=${WATCH_ROOT:-${OUTPUT_ROOT}.watch}
TIMING_PLAN=${TIMING_PLAN:?set the frozen Stage-2 timing plan}
EXECUTION_ADDENDUM=${EXECUTION_ADDENDUM:?set the frozen 2x5090 addendum}
POLL_SECONDS=${POLL_SECONDS:-60}
MAX_WAIT_SECONDS=${MAX_WAIT_SECONDS:-604800}
EXPECTED_HOSTNAME=${EXPECTED_HOSTNAME:-user-Super-Server}
UPSTREAM_COMMIT=be2324829b0e91c8fd10a74d4b43714fde6676e1

if [[ "$(hostname)" != "$EXPECTED_HOSTNAME" ]]; then
  echo "the Stage-2 timing watcher is restricted to hku-gpu2" >&2; exit 2
fi
if [[ -e "$OUTPUT_ROOT" || -e "$WATCH_ROOT" ]]; then
  echo "refusing to reuse Stage-2 timing or watcher output" >&2; exit 3
fi
if [[ "$POLL_SECONDS" -lt 10 || "$MAX_WAIT_SECONDS" -lt "$POLL_SECONDS" ]]; then
  echo "invalid watcher interval/budget" >&2; exit 2
fi
test "$(git -C "$UPSTREAM_REPO" rev-parse HEAD)" = "$UPSTREAM_COMMIT"
test -z "$(git -C "$UPSTREAM_REPO" status --short)"
test "$(git -C "$TOOL_REPO" rev-parse HEAD)" = "$TOOL_COMMIT"
test -z "$(git -C "$TOOL_REPO" status --short)"
test -x "$PYTHON_BIN"
test -s "$TIMING_PLAN"; test -s "$EXECUTION_ADDENDUM"

mkdir -p "$WATCH_ROOT"
sha256sum \
  "$0" "$TOOL_REPO/scripts/run_in_place_ttt_2x5090_stage2_timing.sh" \
  "$TOOL_REPO/scripts/audit_in_place_ttt_2x5090_stage2_timing.py" \
  "$TIMING_PLAN" "$EXECUTION_ADDENDUM" \
  > "$WATCH_ROOT/frozen_inputs.sha256"

write_state() {
  printf '%s\n' "$1" > "$WATCH_ROOT/state.txt.partial"
  mv "$WATCH_ROOT/state.txt.partial" "$WATCH_ROOT/state.txt"
  date --iso-8601=seconds > "$WATCH_ROOT/state_time.txt.partial"
  mv "$WATCH_ROOT/state_time.txt.partial" "$WATCH_ROOT/state_time.txt"
}

snapshot() {
  local prefix=$1 source name
  date --iso-8601=seconds > "$WATCH_ROOT/${prefix}_time.txt"
  for source in "$STAGE2_LADDER_ROOT/state.txt" \
      "${STAGE2_LADDER_ROOT}.watch/state.txt"; do
    name=$(basename "$(dirname "$source")")_$(basename "$source")
    if [[ -f "$source" ]]; then
      cp "$source" "$WATCH_ROOT/${prefix}_${name}"
    else
      : > "$WATCH_ROOT/${prefix}_${name}"
    fi
  done
  tmux list-panes -t "$STAGE2_LADDER_SESSION" \
    -F '#{pane_pid} #{pane_dead} #{pane_current_command}' \
    > "$WATCH_ROOT/${prefix}_stage2_ladder_handle.txt" 2>&1 || true
}

stage2_ladder_valid() {
  "$PYTHON_BIN" - "$STAGE2_LADDER_ROOT/independent_ladder_audit.json" <<'PY'
import json, pathlib, sys
audit = json.loads(pathlib.Path(sys.argv[1]).read_text())
if (audit.get("protocol") != "in_place_ttt_2x5090_stage2_ladder_audit_v1"
        or audit.get("passed") is not True or audit.get("errors") != []
        or set(audit.get("start_models", {})) != {"no_ttt", "ttt"}
        or set(audit.get("boundaries", {})) != {
            "65536_no_ttt", "65536_ttt",
            "131072_no_ttt", "131072_ttt",
        }):
    raise SystemExit(1)
PY
}

started=$(date +%s)
while true; do
  write_state waiting_for_audited_stage2_ladder
  snapshot last
  ladder_state=$(cat "$STAGE2_LADDER_ROOT/state.txt" 2>/dev/null || true)
  if [[ "$ladder_state" == completed_stage2_ladder ]]; then
    if ! stage2_ladder_valid; then
      write_state invalid_stage2_ladder
      exit 4
    fi
    break
  fi
  upstream_state=$(cat "${STAGE2_LADDER_ROOT}.watch/state.txt" 2>/dev/null || true)
  case "$upstream_state" in
    failed_*|invalid_*|timed_out_*|upstream_*|formal_*_missing|stage2_*_missing)
      write_state "upstream_${upstream_state}"
      exit 5
      ;;
  esac
  if ! tmux has-session -t "$STAGE2_LADDER_SESSION" 2>/dev/null; then
    snapshot missing_handle
    write_state stage2_ladder_watcher_handle_missing
    exit 6
  fi
  now=$(date +%s)
  if [[ $((now - started)) -ge "$MAX_WAIT_SECONDS" ]]; then
    write_state timed_out_waiting_for_stage2_ladder
    exit 7
  fi
  sleep "$POLL_SECONDS"
done

write_state launching_stage2_timing
set +e
env \
  UPSTREAM_REPO="$UPSTREAM_REPO" TOOL_REPO="$TOOL_REPO" \
  TOOL_COMMIT="$TOOL_COMMIT" PYTHON_BIN="$PYTHON_BIN" \
  CORPUS_ROOT="$CORPUS_ROOT" STAGE1_ROOT="$STAGE1_ROOT" \
  STAGE2_LADDER_ROOT="$STAGE2_LADDER_ROOT" OUTPUT_ROOT="$OUTPUT_ROOT" \
  TIMING_PLAN="$TIMING_PLAN" EXECUTION_ADDENDUM="$EXECUTION_ADDENDUM" \
  EXPECTED_HOSTNAME="$EXPECTED_HOSTNAME" \
  bash "$TOOL_REPO/scripts/run_in_place_ttt_2x5090_stage2_timing.sh"
status=$?
set -e
printf '%s\n' "$status" > "$WATCH_ROOT/stage2_timing_runner_exit_code.txt"
if [[ "$status" -ne 0 ]]; then
  write_state failed_stage2_timing
  exit 8
fi
if [[ "$(cat "$OUTPUT_ROOT/state.txt" 2>/dev/null || true)" \
      != completed_stage2_timing \
      || ! -s "$OUTPUT_ROOT/independent_audit.json" ]]; then
  write_state invalid_stage2_timing_output
  exit 9
fi
if ! "$PYTHON_BIN" - "$OUTPUT_ROOT/independent_audit.json" <<'PY'
import json, math, pathlib, sys
audit = json.loads(pathlib.Path(sys.argv[1]).read_text())
arms = audit.get("arms", {})
projections = [arms.get(name, {}).get("projected_formal_stage2_seconds")
               for name in ("no_ttt", "ttt")]
if (audit.get("protocol") != "in_place_ttt_2x5090_stage2_timing_audit_v1"
        or audit.get("passed") is not True or audit.get("errors") != []
        or set(arms) != {"no_ttt", "ttt"}
        or not all(isinstance(value, (int, float)) and math.isfinite(value)
                   and value > 0 for value in projections)):
    raise SystemExit(1)
PY
then
  write_state invalid_stage2_timing_audit
  exit 10
fi
write_state completed_stage2_timing
