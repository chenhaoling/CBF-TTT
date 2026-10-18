#!/usr/bin/env bash
set -euo pipefail

UPSTREAM_REPO=${UPSTREAM_REPO:?set the clean official checkout}
TOOL_REPO=${TOOL_REPO:?set the clean CBF-TTT checkout}
TOOL_COMMIT=${TOOL_COMMIT:?set the exact tool commit}
PYTHON_BIN=${PYTHON_BIN:?set the official training Python}
MODEL=${MODEL:?set the Qwen3-4B model directory}
CORPUS_ROOT=${CORPUS_ROOT:?set the independently audited full corpus}
TIMING_ROOT=${TIMING_ROOT:?set the future paired Stage-1 timing root}
TIMING_SESSION=${TIMING_SESSION:?set the Stage-1 timing watcher tmux session}
RELEASE=${RELEASE:?set the future explicit Stage-1 release JSON}
OUTPUT_ROOT=${OUTPUT_ROOT:?set a new paired formal Stage-1 output root}
WATCH_ROOT=${WATCH_ROOT:-${OUTPUT_ROOT}.watch}
FORMAL_PLAN=${FORMAL_PLAN:?set the frozen formal Stage-1 plan}
EXECUTION_ADDENDUM=${EXECUTION_ADDENDUM:?set the frozen 2x5090 addendum}
POLL_SECONDS=${POLL_SECONDS:-60}
MAX_WAIT_SECONDS=${MAX_WAIT_SECONDS:-604800}
EXPECTED_HOSTNAME=${EXPECTED_HOSTNAME:-user-Super-Server}
UPSTREAM_COMMIT=be2324829b0e91c8fd10a74d4b43714fde6676e1

if [[ "$(hostname)" != "$EXPECTED_HOSTNAME" ]]; then
  echo "the formal Stage-1 watcher is restricted to hku-gpu2" >&2; exit 2
fi
if [[ -e "$OUTPUT_ROOT" || -e "${OUTPUT_ROOT}.partial" || -e "$WATCH_ROOT" ]]; then
  echo "refusing to reuse formal Stage-1 or watcher output" >&2; exit 3
fi
if [[ "$POLL_SECONDS" -lt 10 || "$MAX_WAIT_SECONDS" -lt "$POLL_SECONDS" ]]; then
  echo "invalid watcher interval/budget" >&2; exit 2
fi
test "$(git -C "$UPSTREAM_REPO" rev-parse HEAD)" = "$UPSTREAM_COMMIT"
test -z "$(git -C "$UPSTREAM_REPO" status --short)"
test "$(git -C "$TOOL_REPO" rev-parse HEAD)" = "$TOOL_COMMIT"
test -z "$(git -C "$TOOL_REPO" status --short)"
test -x "$PYTHON_BIN"; test -s "$MODEL/config.json"
test -s "$FORMAL_PLAN"; test -s "$EXECUTION_ADDENDUM"

mkdir -p "$WATCH_ROOT"
sha256sum \
  "$0" "$TOOL_REPO/scripts/run_in_place_ttt_2x5090_100m_stage1.sh" \
  "$TOOL_REPO/scripts/audit_in_place_ttt_2x5090_100m_stage1.py" \
  "$TOOL_REPO/scripts/create_in_place_ttt_2x5090_timing_release.py" \
  "$FORMAL_PLAN" "$EXECUTION_ADDENDUM" "$MODEL/config.json" \
  > "$WATCH_ROOT/frozen_inputs.sha256"

write_state() {
  printf '%s\n' "$1" > "$WATCH_ROOT/state.txt.partial"
  mv "$WATCH_ROOT/state.txt.partial" "$WATCH_ROOT/state.txt"
  date --iso-8601=seconds > "$WATCH_ROOT/state_time.txt.partial"
  mv "$WATCH_ROOT/state_time.txt.partial" "$WATCH_ROOT/state_time.txt"
}

snapshot_timing() {
  local prefix=$1
  date --iso-8601=seconds > "$WATCH_ROOT/${prefix}_time.txt"
  if [[ -f "$TIMING_ROOT/state.txt" ]]; then
    cp "$TIMING_ROOT/state.txt" "$WATCH_ROOT/${prefix}_timing_state.txt"
  else
    : > "$WATCH_ROOT/${prefix}_timing_state.txt"
  fi
  tmux list-panes -t "$TIMING_SESSION" \
    -F '#{pane_pid} #{pane_dead} #{pane_current_command}' \
    > "$WATCH_ROOT/${prefix}_timing_handle.txt" 2>&1 || true
}

timing_valid() {
  "$PYTHON_BIN" - "$TIMING_ROOT/independent_audit.json" <<'PY'
import json, math, pathlib, sys
audit = json.loads(pathlib.Path(sys.argv[1]).read_text())
projection = audit.get("projected_sequential_stage1_seconds")
if (audit.get("protocol") != "in_place_ttt_2x5090_stage1_timing_audit_v1"
        or audit.get("passed") is not True or audit.get("errors") != []
        or not isinstance(projection, (int, float))
        or isinstance(projection, bool)
        or not math.isfinite(float(projection)) or projection <= 0):
    raise SystemExit(1)
PY
}

release_valid() {
  "$PYTHON_BIN" - "$TIMING_ROOT/independent_audit.json" "$RELEASE" <<'PY'
import hashlib, json, pathlib, sys
timing_path, release_path = map(pathlib.Path, sys.argv[1:])
digest = hashlib.sha256(timing_path.read_bytes()).hexdigest()
release = json.loads(release_path.read_text())
if (release.get("protocol") != "in_place_ttt_2x5090_100m_stage1_release_v1"
        or release.get("accepted") is not True
        or release.get("stage") != "stage1"
        or release.get("timing_audit_sha256") != digest
        or not str(release.get("reviewed_by", "")).strip()
        or not str(release.get("decision_note", "")).strip()):
    raise SystemExit(1)
PY
}

started=$(date +%s)
while true; do
  write_state waiting_for_audited_stage1_timing
  snapshot_timing last
  timing_state=$(cat "$TIMING_ROOT/state.txt" 2>/dev/null || true)
  case "$timing_state" in
    completed_stage1_timing)
      if ! timing_valid; then
        write_state invalid_stage1_timing
        exit 4
      fi
      break
      ;;
    failed_*|invalid_*|timed_out_*|upstream_*)
      write_state "upstream_${timing_state}"
      exit 5
      ;;
  esac
  if ! tmux has-session -t "$TIMING_SESSION" 2>/dev/null; then
    snapshot_timing missing_handle
    write_state stage1_timing_watcher_handle_missing
    exit 6
  fi
  now=$(date +%s)
  if [[ $((now - started)) -ge "$MAX_WAIT_SECONDS" ]]; then
    write_state timed_out_waiting_for_stage1_timing
    exit 7
  fi
  sleep "$POLL_SECONDS"
done

while true; do
  write_state waiting_for_explicit_stage1_release
  date --iso-8601=seconds > "$WATCH_ROOT/last_release_time.txt"
  if [[ -s "$RELEASE" ]]; then
    if ! timing_valid || ! release_valid; then
      write_state invalid_explicit_stage1_release
      exit 8
    fi
    sha256sum "$TIMING_ROOT/independent_audit.json" "$RELEASE" \
      > "$WATCH_ROOT/accepted_release.sha256"
    break
  fi
  if [[ -e "${RELEASE}.partial" ]]; then
    printf '%s\n' present > "$WATCH_ROOT/last_release_partial.txt"
  else
    printf '%s\n' absent > "$WATCH_ROOT/last_release_partial.txt"
  fi
  now=$(date +%s)
  if [[ $((now - started)) -ge "$MAX_WAIT_SECONDS" ]]; then
    write_state timed_out_waiting_for_explicit_stage1_release
    exit 9
  fi
  sleep "$POLL_SECONDS"
done

write_state launching_formal_stage1
set +e
env \
  UPSTREAM_REPO="$UPSTREAM_REPO" TOOL_REPO="$TOOL_REPO" \
  TOOL_COMMIT="$TOOL_COMMIT" PYTHON_BIN="$PYTHON_BIN" MODEL="$MODEL" \
  CORPUS_ROOT="$CORPUS_ROOT" TIMING_ROOT="$TIMING_ROOT" RELEASE="$RELEASE" \
  OUTPUT_ROOT="$OUTPUT_ROOT" FORMAL_PLAN="$FORMAL_PLAN" \
  EXECUTION_ADDENDUM="$EXECUTION_ADDENDUM" \
  EXPECTED_HOSTNAME="$EXPECTED_HOSTNAME" \
  bash "$TOOL_REPO/scripts/run_in_place_ttt_2x5090_100m_stage1.sh"
status=$?
set -e
printf '%s\n' "$status" > "$WATCH_ROOT/formal_runner_exit_code.txt"
if [[ "$status" -ne 0 ]]; then
  write_state failed_formal_stage1
  exit 10
fi
if [[ "$(cat "$OUTPUT_ROOT/state.txt" 2>/dev/null || true)" \
      != completed_100m_stage1 || ! -s "$OUTPUT_ROOT/independent_audit.json" ]]; then
  write_state invalid_formal_stage1_output
  exit 11
fi
"$PYTHON_BIN" - "$OUTPUT_ROOT/independent_audit.json" <<'PY'
import json, pathlib, sys
audit = json.loads(pathlib.Path(sys.argv[1]).read_text())
if (audit.get("protocol") != "in_place_ttt_2x5090_100m_stage1_audit_v1"
        or audit.get("passed") is not True or audit.get("errors") != []):
    raise SystemExit(1)
PY
write_state completed_formal_stage1
