#!/usr/bin/env bash
set -euo pipefail

UPSTREAM_REPO=${UPSTREAM_REPO:?set the clean official checkout}
TOOL_REPO=${TOOL_REPO:?set the clean CBF-TTT checkout}
TOOL_COMMIT=${TOOL_COMMIT:?set the exact tool commit}
PYTHON_BIN=${PYTHON_BIN:?set the baseline training Python}
MODEL_IDENTITY=${MODEL_IDENTITY:?set the passing Base identity JSON}
STAGE1_ROOT=${STAGE1_ROOT:?set the corrected formal Stage-1 result root}
STAGE1_WATCH_ROOT=${STAGE1_WATCH_ROOT:?set the corrected Stage-1 watcher root}
STAGE1_SESSION=${STAGE1_SESSION:?set the corrected Stage-1 tmux session}
RESOURCE_INPUT_ROOT=${RESOURCE_INPUT_ROOT:?set the future corrected inputs root}
OUTPUT_ROOT=${OUTPUT_ROOT:?set a new corrected Stage-2 ladder root}
WATCH_ROOT=${WATCH_ROOT:-${OUTPUT_ROOT}.watch}
BASELINE_PLAN=${BASELINE_PLAN:?set the parent baseline plan}
RESOURCE_PLAN=${RESOURCE_PLAN:?set the corrected Stage-2 resource plan}
EXECUTION_ADDENDUM=${EXECUTION_ADDENDUM:?set the corrected formal Stage-1 plan}
INPUT_ADDENDUM=${INPUT_ADDENDUM:?set the corrected input addendum}
POLL_SECONDS=${POLL_SECONDS:-60}
MAX_WAIT_SECONDS=${MAX_WAIT_SECONDS:-1209600}
EXPECTED_HOSTNAME=${EXPECTED_HOSTNAME:-user-Super-Server}
UPSTREAM_COMMIT=be2324829b0e91c8fd10a74d4b43714fde6676e1
RUNNER="$TOOL_REPO/scripts/run_in_place_ttt_2x5090_stage2_ladder.sh"
UNDERLYING_AUDITOR="$TOOL_REPO/scripts/audit_in_place_ttt_2x5090_stage2_ladder.py"
WRAPPER_AUDITOR="$TOOL_REPO/scripts/audit_in_place_ttt_qwen3_4b_base_stage2_ladder.py"
CORRECTED_STAGE1_AUDIT="$STAGE1_WATCH_ROOT/corrected_independent_audit.json"
NO_TTT_CONFIG="$TOOL_REPO/configs/baseline_reproduction/qwen3_4b_2x5090_100m_stage2_no_ttt.yaml"
TTT_CONFIG="$TOOL_REPO/configs/baseline_reproduction/qwen3_4b_2x5090_100m_stage2_ttt.yaml"

if [[ "$(hostname)" != "$EXPECTED_HOSTNAME" ]]; then
  echo "corrected Stage-2 ladder is restricted to hku-gpu2" >&2; exit 2
fi
if [[ -e "$OUTPUT_ROOT" || -e "$WATCH_ROOT" ]]; then
  echo "refusing to reuse corrected Stage-2 ladder roots" >&2; exit 3
fi
if [[ "$POLL_SECONDS" -lt 10 || "$MAX_WAIT_SECONDS" -lt "$POLL_SECONDS" ]]; then
  echo "invalid watcher interval/budget" >&2; exit 2
fi
test "$(git -C "$UPSTREAM_REPO" rev-parse HEAD)" = "$UPSTREAM_COMMIT"
test -z "$(git -C "$UPSTREAM_REPO" status --short)"
test "$(git -C "$TOOL_REPO" rev-parse HEAD)" = "$TOOL_COMMIT"
test -z "$(git -C "$TOOL_REPO" status --short)"
for path in "$PYTHON_BIN" "$MODEL_IDENTITY" "$BASELINE_PLAN" \
  "$RESOURCE_PLAN" "$EXECUTION_ADDENDUM" "$INPUT_ADDENDUM" "$RUNNER" \
  "$UNDERLYING_AUDITOR" "$WRAPPER_AUDITOR" "$NO_TTT_CONFIG" \
  "$TTT_CONFIG"; do test -s "$path"; done

mkdir -p "$WATCH_ROOT"
sha256sum "$0" "$RUNNER" "$UNDERLYING_AUDITOR" "$WRAPPER_AUDITOR" \
  "$MODEL_IDENTITY" "$BASELINE_PLAN" "$RESOURCE_PLAN" \
  "$EXECUTION_ADDENDUM" "$INPUT_ADDENDUM" "$NO_TTT_CONFIG" \
  "$TTT_CONFIG" > "$WATCH_ROOT/frozen_inputs.sha256"

write_state() {
  printf '%s\n' "$1" > "$WATCH_ROOT/state.txt.partial"
  mv "$WATCH_ROOT/state.txt.partial" "$WATCH_ROOT/state.txt"
  date --iso-8601=seconds > "$WATCH_ROOT/state_time.txt.partial"
  mv "$WATCH_ROOT/state_time.txt.partial" "$WATCH_ROOT/state_time.txt"
}

snapshot() {
  local prefix=$1
  date --iso-8601=seconds > "$WATCH_ROOT/${prefix}_time.txt"
  if [[ -f "$STAGE1_WATCH_ROOT/state.txt" ]]; then
    cp "$STAGE1_WATCH_ROOT/state.txt" "$WATCH_ROOT/${prefix}_stage1_state.txt"
  else
    : > "$WATCH_ROOT/${prefix}_stage1_state.txt"
  fi
  tmux list-panes -t "$STAGE1_SESSION" \
    -F '#{pane_pid} #{pane_dead} #{pane_current_command}' \
    > "$WATCH_ROOT/${prefix}_stage1_handle.txt" 2>&1 || true
}

stage1_valid() {
  "$PYTHON_BIN" - "$STAGE1_ROOT/independent_audit.json" \
    "$CORRECTED_STAGE1_AUDIT" <<'PY'
import hashlib, json, pathlib, sys
underlying_path, corrected_path = map(pathlib.Path, sys.argv[1:])
underlying = json.loads(underlying_path.read_text())
corrected = json.loads(corrected_path.read_text())
digest = hashlib.sha256(underlying_path.read_bytes()).hexdigest()
if (underlying.get("protocol")
        != "in_place_ttt_2x5090_100m_stage1_audit_v1"
        or underlying.get("passed") is not True or underlying.get("errors") != []
        or set(underlying.get("arms", {})) != {"no_ttt", "ttt"}
        or corrected.get("protocol")
        != "in_place_ttt_qwen3_4b_base_100m_stage1_audit_v1"
        or corrected.get("passed") is not True or corrected.get("errors") != []
        or corrected.get("underlying_formal_audit_sha256") != digest):
    raise SystemExit("corrected formal Stage-1 chain is invalid")
PY
}

started=$(date +%s)
while true; do
  write_state waiting_for_corrected_formal_stage1
  snapshot last
  stage1_state=$(cat "$STAGE1_WATCH_ROOT/state.txt" 2>/dev/null || true)
  case "$stage1_state" in
    completed_corrected_formal_stage1)
      test "$(cat "$STAGE1_ROOT/state.txt")" = completed_100m_stage1
      test -s "$STAGE1_ROOT/independent_audit.json"
      test -s "$CORRECTED_STAGE1_AUDIT"
      stage1_valid
      test -s "$RESOURCE_INPUT_ROOT/manifest.json"
      test -s "$RESOURCE_INPUT_ROOT/independent_audit.json"
      sha256sum "$STAGE1_ROOT/independent_audit.json" \
        "$CORRECTED_STAGE1_AUDIT" "$RESOURCE_INPUT_ROOT/manifest.json" \
        "$RESOURCE_INPUT_ROOT/independent_audit.json" \
        > "$WATCH_ROOT/released_inputs.sha256"
      break
      ;;
    failed_*|invalid_*|timed_out_*|upstream_*|corrected_stage1_timing_handle_missing)
      write_state "upstream_${stage1_state}"
      exit 4
      ;;
  esac
  if ! tmux has-session -t "$STAGE1_SESSION" 2>/dev/null; then
    snapshot missing_handle
    write_state corrected_formal_stage1_handle_missing
    exit 5
  fi
  now=$(date +%s)
  [[ $((now - started)) -lt "$MAX_WAIT_SECONDS" ]] || {
    write_state timed_out_waiting_for_corrected_formal_stage1; exit 6;
  }
  sleep "$POLL_SECONDS"
done

write_state launching_corrected_stage2_ladder
set +e
env UPSTREAM_REPO="$UPSTREAM_REPO" PYTHON_BIN="$PYTHON_BIN" \
  STAGE1_ROOT="$STAGE1_ROOT" RESOURCE_INPUT_ROOT="$RESOURCE_INPUT_ROOT" \
  OUTPUT_ROOT="$OUTPUT_ROOT" BASELINE_PLAN="$BASELINE_PLAN" \
  RESOURCE_PLAN="$RESOURCE_PLAN" EXECUTION_ADDENDUM="$EXECUTION_ADDENDUM" \
  INPUT_ADDENDUM="$INPUT_ADDENDUM" TOOL_REPO="$TOOL_REPO" \
  TOOL_COMMIT="$TOOL_COMMIT" EXPECTED_HOSTNAME="$EXPECTED_HOSTNAME" \
  bash "$RUNNER"
status=$?
set -e
printf '%s\n' "$status" > "$WATCH_ROOT/runner_exit_code.txt"
[[ "$status" -eq 0 ]] || { write_state failed_corrected_stage2_ladder; exit 7; }

write_state auditing_corrected_stage2_ladder
"$PYTHON_BIN" "$WRAPPER_AUDITOR" \
  --root "$OUTPUT_ROOT" --stage1-root "$STAGE1_ROOT" \
  --corrected-stage1-audit "$CORRECTED_STAGE1_AUDIT" \
  --model-identity "$MODEL_IDENTITY" \
  --resource-input-root "$RESOURCE_INPUT_ROOT" \
  --resource-plan "$RESOURCE_PLAN" \
  --execution-addendum "$EXECUTION_ADDENDUM" \
  --input-addendum "$INPUT_ADDENDUM" --runner "$RUNNER" \
  --underlying-auditor "$UNDERLYING_AUDITOR" \
  --no-ttt-config "$NO_TTT_CONFIG" --ttt-config "$TTT_CONFIG" \
  --output "$WATCH_ROOT/corrected_independent_audit.json" \
  > "$WATCH_ROOT/corrected_independent_audit.stdout"
write_state completed_corrected_stage2_ladder
