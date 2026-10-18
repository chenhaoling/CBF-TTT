#!/usr/bin/env bash
set -euo pipefail

UPSTREAM_REPO=${UPSTREAM_REPO:?set the clean official checkout}
TOOL_REPO=${TOOL_REPO:?set the clean CBF-TTT checkout}
TOOL_COMMIT=${TOOL_COMMIT:?set the exact tool commit}
PYTHON_BIN=${PYTHON_BIN:?set the official training Python}
MODEL_IDENTITY=${MODEL_IDENTITY:?set the passing Base identity JSON}
CORPUS_ROOT=${CORPUS_ROOT:?set the independently audited Base full corpus}
CORPUS_AUDIT=${CORPUS_AUDIT:?set the Base full-corpus audit}
STAGE1_ROOT=${STAGE1_ROOT:?set the corrected formal Stage-1 root}
CORRECTED_STAGE1_AUDIT=${CORRECTED_STAGE1_AUDIT:?set the corrected Stage-1 audit}
LADDER_ROOT=${LADDER_ROOT:?set the corrected Stage-2 ladder result root}
LADDER_WATCH_ROOT=${LADDER_WATCH_ROOT:?set the corrected ladder watcher root}
LADDER_SESSION=${LADDER_SESSION:?set the corrected ladder tmux session}
OUTPUT_ROOT=${OUTPUT_ROOT:?set a new timing output root}
WATCH_ROOT=${WATCH_ROOT:-${OUTPUT_ROOT}.watch}
TIMING_PLAN=${TIMING_PLAN:?set the corrected Base Stage-2 timing plan}
RESOURCE_PLAN=${RESOURCE_PLAN:?set the corrected Base Stage-2 resource plan}
POLL_SECONDS=${POLL_SECONDS:-60}
MAX_WAIT_SECONDS=${MAX_WAIT_SECONDS:-604800}
EXPECTED_HOSTNAME=${EXPECTED_HOSTNAME:-user-Super-Server}
UPSTREAM_COMMIT=be2324829b0e91c8fd10a74d4b43714fde6676e1
RUNNER="$TOOL_REPO/scripts/run_in_place_ttt_2x5090_stage2_timing.sh"
UNDERLYING_AUDITOR="$TOOL_REPO/scripts/audit_in_place_ttt_2x5090_stage2_timing.py"
WRAPPER_AUDITOR="$TOOL_REPO/scripts/audit_in_place_ttt_qwen3_4b_base_stage2_timing.py"
LADDER_AUDIT="$LADDER_ROOT/independent_ladder_audit.json"
CORRECTED_LADDER_AUDIT="$LADDER_WATCH_ROOT/corrected_independent_audit.json"
NO_TTT_CONFIG="$TOOL_REPO/configs/baseline_reproduction/qwen3_4b_2x5090_100m_stage2_no_ttt.yaml"
TTT_CONFIG="$TOOL_REPO/configs/baseline_reproduction/qwen3_4b_2x5090_100m_stage2_ttt.yaml"

if [[ "$(hostname)" != "$EXPECTED_HOSTNAME" ]]; then
  echo "corrected Base Stage-2 timing is restricted to hku-gpu2" >&2; exit 2
fi
if [[ -e "$OUTPUT_ROOT" || -e "${OUTPUT_ROOT}.partial" || -e "$WATCH_ROOT" ]]; then
  echo "refusing to reuse corrected timing roots" >&2; exit 3
fi
if [[ "$POLL_SECONDS" -lt 10 || "$MAX_WAIT_SECONDS" -lt "$POLL_SECONDS" ]]; then
  echo "invalid watcher interval/budget" >&2; exit 2
fi
test "$(git -C "$UPSTREAM_REPO" rev-parse HEAD)" = "$UPSTREAM_COMMIT"
test -z "$(git -C "$UPSTREAM_REPO" status --short)"
test "$(git -C "$TOOL_REPO" rev-parse HEAD)" = "$TOOL_COMMIT"
test -z "$(git -C "$TOOL_REPO" status --short)"
for path in "$PYTHON_BIN" "$MODEL_IDENTITY" "$TIMING_PLAN" \
  "$RESOURCE_PLAN" "$RUNNER" "$UNDERLYING_AUDITOR" "$WRAPPER_AUDITOR" \
  "$NO_TTT_CONFIG" "$TTT_CONFIG"; do test -s "$path"; done

mkdir -p "$WATCH_ROOT"
sha256sum "$0" "$RUNNER" "$UNDERLYING_AUDITOR" "$WRAPPER_AUDITOR" \
  "$MODEL_IDENTITY" "$TIMING_PLAN" "$RESOURCE_PLAN" "$NO_TTT_CONFIG" \
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
  for name in state.txt state_time.txt; do
    if [[ -f "$LADDER_WATCH_ROOT/$name" ]]; then
      cp "$LADDER_WATCH_ROOT/$name" "$WATCH_ROOT/${prefix}_ladder_${name}"
    else
      : > "$WATCH_ROOT/${prefix}_ladder_${name}"
    fi
  done
  tmux list-panes -t "$LADDER_SESSION" \
    -F '#{pane_pid} #{pane_dead} #{pane_current_command}' \
    > "$WATCH_ROOT/${prefix}_ladder_handle.txt" 2>&1 || true
}

ladder_valid() {
  "$PYTHON_BIN" - "$LADDER_AUDIT" "$CORRECTED_LADDER_AUDIT" \
    "$CORRECTED_STAGE1_AUDIT" <<'PY'
import hashlib, json, pathlib, sys
underlying_path, corrected_path, stage1_path = map(pathlib.Path, sys.argv[1:])
underlying = json.loads(underlying_path.read_text())
corrected = json.loads(corrected_path.read_text())
if (underlying.get("protocol") != "in_place_ttt_2x5090_stage2_ladder_audit_v1"
        or underlying.get("passed") is not True or underlying.get("errors") != []
        or set(underlying.get("boundaries", {})) != {
            "65536_no_ttt", "65536_ttt", "131072_no_ttt", "131072_ttt"}
        or corrected.get("protocol")
        != "in_place_ttt_qwen3_4b_base_stage2_ladder_audit_v1"
        or corrected.get("passed") is not True or corrected.get("errors") != []
        or corrected.get("underlying_ladder_audit_sha256")
        != hashlib.sha256(underlying_path.read_bytes()).hexdigest()
        or corrected.get("corrected_stage1_audit_sha256")
        != hashlib.sha256(stage1_path.read_bytes()).hexdigest()):
    raise SystemExit("corrected Stage-2 ladder chain is invalid")
PY
}

started=$(date +%s)
while true; do
  write_state waiting_for_corrected_stage2_ladder
  snapshot last
  ladder_watch_state=$(cat "$LADDER_WATCH_ROOT/state.txt" 2>/dev/null || true)
  case "$ladder_watch_state" in
    completed_corrected_stage2_ladder)
      test "$(cat "$LADDER_ROOT/state.txt")" = completed_stage2_ladder
      test -s "$LADDER_AUDIT"
      test -s "$CORRECTED_LADDER_AUDIT"
      test -s "$CORRECTED_STAGE1_AUDIT"
      ladder_valid
      test -s "$CORPUS_ROOT/manifest.json"
      test -s "$CORPUS_ROOT/stage2_128k.jsonl"
      test -s "$CORPUS_AUDIT"
      sha256sum "$CORPUS_ROOT/manifest.json" "$CORPUS_ROOT/stage2_128k.jsonl" \
        "$CORPUS_AUDIT" "$CORRECTED_STAGE1_AUDIT" "$LADDER_AUDIT" \
        "$CORRECTED_LADDER_AUDIT" > "$WATCH_ROOT/released_inputs.sha256"
      break
      ;;
    failed_*|invalid_*|timed_out_*|upstream_*|corrected_formal_stage1_handle_missing)
      write_state "upstream_${ladder_watch_state}"
      exit 4
      ;;
  esac
  if ! tmux has-session -t "$LADDER_SESSION" 2>/dev/null; then
    snapshot missing_handle
    write_state corrected_stage2_ladder_handle_missing
    exit 5
  fi
  now=$(date +%s)
  if [[ $((now - started)) -ge "$MAX_WAIT_SECONDS" ]]; then
    write_state timed_out_waiting_for_corrected_stage2_ladder
    exit 6
  fi
  sleep "$POLL_SECONDS"
done

write_state launching_corrected_stage2_timing
set +e
env UPSTREAM_REPO="$UPSTREAM_REPO" TOOL_REPO="$TOOL_REPO" \
  TOOL_COMMIT="$TOOL_COMMIT" PYTHON_BIN="$PYTHON_BIN" \
  CORPUS_ROOT="$CORPUS_ROOT" STAGE1_ROOT="$STAGE1_ROOT" \
  STAGE2_LADDER_ROOT="$LADDER_ROOT" OUTPUT_ROOT="$OUTPUT_ROOT" \
  TIMING_PLAN="$TIMING_PLAN" EXECUTION_ADDENDUM="$RESOURCE_PLAN" \
  EXPECTED_HOSTNAME="$EXPECTED_HOSTNAME" bash "$RUNNER"
status=$?
set -e
printf '%s\n' "$status" > "$WATCH_ROOT/timing_runner_exit_code.txt"
if [[ "$status" -ne 0 ]]; then
  write_state failed_corrected_stage2_timing
  exit 7
fi

write_state auditing_corrected_stage2_timing
"$PYTHON_BIN" "$WRAPPER_AUDITOR" \
  --timing-root "$OUTPUT_ROOT" --stage1-root "$STAGE1_ROOT" \
  --corrected-stage1-audit "$CORRECTED_STAGE1_AUDIT" \
  --ladder-root "$LADDER_ROOT" \
  --corrected-ladder-audit "$CORRECTED_LADDER_AUDIT" \
  --model-identity "$MODEL_IDENTITY" --corpus-root "$CORPUS_ROOT" \
  --corpus-audit "$CORPUS_AUDIT" --timing-plan "$TIMING_PLAN" \
  --resource-plan "$RESOURCE_PLAN" --runner "$RUNNER" \
  --underlying-auditor "$UNDERLYING_AUDITOR" \
  --no-ttt-config "$NO_TTT_CONFIG" --ttt-config "$TTT_CONFIG" \
  --output "$WATCH_ROOT/corrected_independent_audit.json" \
  > "$WATCH_ROOT/corrected_independent_audit.stdout"
write_state completed_corrected_stage2_timing
