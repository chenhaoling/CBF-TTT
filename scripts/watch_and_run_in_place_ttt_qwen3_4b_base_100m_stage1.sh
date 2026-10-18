#!/usr/bin/env bash
set -euo pipefail

UPSTREAM_REPO=${UPSTREAM_REPO:?set the clean official checkout}
TOOL_REPO=${TOOL_REPO:?set the clean CBF-TTT checkout}
TOOL_COMMIT=${TOOL_COMMIT:?set the exact tool commit}
PYTHON_BIN=${PYTHON_BIN:?set the official training Python}
MODEL=${MODEL:?set the exact Qwen3-4B-Base directory}
MODEL_IDENTITY=${MODEL_IDENTITY:?set the passing Base identity JSON}
CORPUS_ROOT=${CORPUS_ROOT:?set the corrected full corpus root}
TIMING_ROOT=${TIMING_ROOT:?set the corrected timing result root}
TIMING_WATCH_ROOT=${TIMING_WATCH_ROOT:?set the corrected timing watcher root}
TIMING_SESSION=${TIMING_SESSION:?set the corrected timing tmux session}
RELEASE=${RELEASE:?set the future explicit corrected Stage-1 release JSON}
OUTPUT_ROOT=${OUTPUT_ROOT:?set a new formal Stage-1 output root}
WATCH_ROOT=${WATCH_ROOT:-${OUTPUT_ROOT}.watch}
FORMAL_PLAN=${FORMAL_PLAN:?set the corrected formal Stage-1 plan}
EXECUTION_ADDENDUM=${EXECUTION_ADDENDUM:?set the corrected timing addendum}
POLL_SECONDS=${POLL_SECONDS:-60}
MAX_WAIT_SECONDS=${MAX_WAIT_SECONDS:-1209600}
EXPECTED_HOSTNAME=${EXPECTED_HOSTNAME:-user-Super-Server}
UPSTREAM_COMMIT=be2324829b0e91c8fd10a74d4b43714fde6676e1
RUNNER="$TOOL_REPO/scripts/run_in_place_ttt_2x5090_100m_stage1.sh"
UNDERLYING_AUDITOR="$TOOL_REPO/scripts/audit_in_place_ttt_2x5090_100m_stage1.py"
WRAPPER_AUDITOR="$TOOL_REPO/scripts/audit_in_place_ttt_qwen3_4b_base_100m_stage1.py"
RELEASE_TOOL="$TOOL_REPO/scripts/create_in_place_ttt_qwen3_4b_base_stage1_release.py"
NO_TTT_CONFIG="$TOOL_REPO/configs/baseline_reproduction/qwen3_4b_2x5090_100m_stage1_no_ttt.yaml"
TTT_CONFIG="$TOOL_REPO/configs/baseline_reproduction/qwen3_4b_2x5090_100m_stage1_ttt.yaml"
CORRECTED_TIMING_AUDIT="$TIMING_WATCH_ROOT/corrected_independent_audit.json"

if [[ "$(hostname)" != "$EXPECTED_HOSTNAME" ]]; then
  echo "corrected formal Stage-1 is restricted to hku-gpu2" >&2; exit 2
fi
if [[ -e "$OUTPUT_ROOT" || -e "${OUTPUT_ROOT}.partial" || -e "$WATCH_ROOT" ]]; then
  echo "refusing to reuse corrected formal Stage-1 roots" >&2; exit 3
fi
if [[ "$POLL_SECONDS" -lt 10 || "$MAX_WAIT_SECONDS" -lt "$POLL_SECONDS" ]]; then
  echo "invalid watcher interval/budget" >&2; exit 2
fi
test "$(git -C "$UPSTREAM_REPO" rev-parse HEAD)" = "$UPSTREAM_COMMIT"
test -z "$(git -C "$UPSTREAM_REPO" status --short)"
test "$(git -C "$TOOL_REPO" rev-parse HEAD)" = "$TOOL_COMMIT"
test -z "$(git -C "$TOOL_REPO" status --short)"
for path in "$PYTHON_BIN" "$MODEL/config.json" "$MODEL_IDENTITY" \
  "$FORMAL_PLAN" "$EXECUTION_ADDENDUM" "$RUNNER" "$UNDERLYING_AUDITOR" \
  "$WRAPPER_AUDITOR" "$RELEASE_TOOL" "$NO_TTT_CONFIG" "$TTT_CONFIG"; do
  test -s "$path"
done

mkdir -p "$WATCH_ROOT"
sha256sum "$0" "$RUNNER" "$UNDERLYING_AUDITOR" "$WRAPPER_AUDITOR" \
  "$RELEASE_TOOL" "$MODEL/config.json" "$MODEL_IDENTITY" "$FORMAL_PLAN" \
  "$EXECUTION_ADDENDUM" "$NO_TTT_CONFIG" "$TTT_CONFIG" \
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
  if [[ -f "$TIMING_WATCH_ROOT/state.txt" ]]; then
    cp "$TIMING_WATCH_ROOT/state.txt" "$WATCH_ROOT/${prefix}_timing_state.txt"
  else
    : > "$WATCH_ROOT/${prefix}_timing_state.txt"
  fi
  tmux list-panes -t "$TIMING_SESSION" \
    -F '#{pane_pid} #{pane_dead} #{pane_current_command}' \
    > "$WATCH_ROOT/${prefix}_timing_handle.txt" 2>&1 || true
}

timing_valid() {
  "$PYTHON_BIN" - "$TIMING_ROOT/independent_audit.json" \
    "$CORRECTED_TIMING_AUDIT" <<'PY'
import hashlib, json, math, pathlib, sys
underlying_path, corrected_path = map(pathlib.Path, sys.argv[1:])
underlying = json.loads(underlying_path.read_text())
corrected = json.loads(corrected_path.read_text())
sha = hashlib.sha256(underlying_path.read_bytes()).hexdigest()
projection = underlying.get("projected_sequential_stage1_seconds")
if (underlying.get("protocol")
        != "in_place_ttt_2x5090_stage1_timing_audit_v1"
        or underlying.get("passed") is not True or underlying.get("errors") != []
        or not isinstance(projection, (int, float)) or isinstance(projection, bool)
        or not math.isfinite(float(projection)) or projection <= 0
        or corrected.get("protocol")
        != "in_place_ttt_qwen3_4b_base_stage1_timing_audit_v1"
        or corrected.get("passed") is not True or corrected.get("errors") != []
        or corrected.get("underlying_timing_audit_sha256") != sha
        or corrected.get("formal_release_created") is not False):
    raise SystemExit("corrected timing chain is invalid")
PY
}

release_valid() {
  "$PYTHON_BIN" - "$TIMING_ROOT/independent_audit.json" \
    "$CORRECTED_TIMING_AUDIT" "$RELEASE" <<'PY'
import hashlib, json, pathlib, sys
timing_path, corrected_path, release_path = map(pathlib.Path, sys.argv[1:])
release = json.loads(release_path.read_text())
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
if (release.get("protocol") != "in_place_ttt_2x5090_100m_stage1_release_v1"
        or release.get("accepted") is not True or release.get("stage") != "stage1"
        or release.get("timing_audit_sha256") != sha(timing_path)
        or release.get("corrected_timing_audit_sha256") != sha(corrected_path)
        or release.get("model_revision")
        != "906bfd4b4dc7f14ee4320094d8b41684abff8539"
        or not str(release.get("reviewed_by", "")).strip()
        or not str(release.get("decision_note", "")).strip()):
    raise SystemExit("corrected explicit release is invalid")
PY
}

started=$(date +%s)
while true; do
  write_state waiting_for_corrected_stage1_timing
  snapshot_timing last
  timing_state=$(cat "$TIMING_WATCH_ROOT/state.txt" 2>/dev/null || true)
  case "$timing_state" in
    completed_corrected_stage1_timing)
      test "$(cat "$TIMING_ROOT/state.txt")" = completed_stage1_timing
      test -s "$TIMING_ROOT/independent_audit.json"
      test -s "$CORRECTED_TIMING_AUDIT"
      timing_valid
      sha256sum "$TIMING_ROOT/independent_audit.json" \
        "$CORRECTED_TIMING_AUDIT" > "$WATCH_ROOT/accepted_timing.sha256"
      break
      ;;
    failed_*|invalid_*|timed_out_*|upstream_*|corrected_stage1_ladder_handle_missing)
      write_state "upstream_${timing_state}"
      exit 4
      ;;
  esac
  if ! tmux has-session -t "$TIMING_SESSION" 2>/dev/null; then
    snapshot_timing missing_handle
    write_state corrected_stage1_timing_handle_missing
    exit 5
  fi
  now=$(date +%s)
  [[ $((now - started)) -lt "$MAX_WAIT_SECONDS" ]] || {
    write_state timed_out_waiting_for_corrected_stage1_timing; exit 6;
  }
  sleep "$POLL_SECONDS"
done

while true; do
  write_state waiting_for_explicit_corrected_stage1_release
  date --iso-8601=seconds > "$WATCH_ROOT/last_release_time.txt"
  if [[ -s "$RELEASE" ]]; then
    timing_valid
    release_valid
    sha256sum "$TIMING_ROOT/independent_audit.json" \
      "$CORRECTED_TIMING_AUDIT" "$RELEASE" \
      > "$WATCH_ROOT/accepted_release.sha256"
    break
  fi
  if [[ -e "${RELEASE}.partial" ]]; then
    printf '%s\n' present > "$WATCH_ROOT/last_release_partial.txt"
  else
    printf '%s\n' absent > "$WATCH_ROOT/last_release_partial.txt"
  fi
  now=$(date +%s)
  [[ $((now - started)) -lt "$MAX_WAIT_SECONDS" ]] || {
    write_state timed_out_waiting_for_explicit_corrected_stage1_release; exit 7;
  }
  sleep "$POLL_SECONDS"
done

write_state launching_corrected_formal_stage1
set +e
env UPSTREAM_REPO="$UPSTREAM_REPO" TOOL_REPO="$TOOL_REPO" \
  TOOL_COMMIT="$TOOL_COMMIT" PYTHON_BIN="$PYTHON_BIN" MODEL="$MODEL" \
  CORPUS_ROOT="$CORPUS_ROOT" TIMING_ROOT="$TIMING_ROOT" RELEASE="$RELEASE" \
  OUTPUT_ROOT="$OUTPUT_ROOT" FORMAL_PLAN="$FORMAL_PLAN" \
  EXECUTION_ADDENDUM="$EXECUTION_ADDENDUM" \
  EXPECTED_HOSTNAME="$EXPECTED_HOSTNAME" bash "$RUNNER"
status=$?
set -e
printf '%s\n' "$status" > "$WATCH_ROOT/formal_runner_exit_code.txt"
[[ "$status" -eq 0 ]] || { write_state failed_corrected_formal_stage1; exit 8; }

write_state auditing_corrected_formal_stage1
"$PYTHON_BIN" "$WRAPPER_AUDITOR" \
  --root "$OUTPUT_ROOT" --model "$MODEL" --model-identity "$MODEL_IDENTITY" \
  --corpus-root "$CORPUS_ROOT" \
  --corpus-audit "$CORPUS_ROOT/independent_audit.json" \
  --timing-root "$TIMING_ROOT" \
  --corrected-timing-audit "$CORRECTED_TIMING_AUDIT" \
  --release "$RELEASE" --formal-plan "$FORMAL_PLAN" \
  --execution-addendum "$EXECUTION_ADDENDUM" --runner "$RUNNER" \
  --underlying-auditor "$UNDERLYING_AUDITOR" \
  --no-ttt-config "$NO_TTT_CONFIG" --ttt-config "$TTT_CONFIG" \
  --output "$WATCH_ROOT/corrected_independent_audit.json" \
  > "$WATCH_ROOT/corrected_independent_audit.stdout"
write_state completed_corrected_formal_stage1
