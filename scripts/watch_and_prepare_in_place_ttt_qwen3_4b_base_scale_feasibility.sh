#!/usr/bin/env bash
set -euo pipefail

TOOL_REPO=${TOOL_REPO:?set the clean CBF-TTT checkout}
TOOL_COMMIT=${TOOL_COMMIT:?set the exact tool commit}
PYTHON_BIN=${PYTHON_BIN:?set the isolated Python}
STAGE1_WATCH_ROOT=${STAGE1_WATCH_ROOT:?set the corrected Stage-1 timing watcher root}
STAGE1_SESSION=${STAGE1_SESSION:?set its tmux session}
STAGE2_WATCH_ROOT=${STAGE2_WATCH_ROOT:?set the corrected Stage-2 timing watcher root}
STAGE2_SESSION=${STAGE2_SESSION:?set its tmux session}
OUTPUT_ROOT=${OUTPUT_ROOT:?set a new scale-feasibility report root}
POLL_SECONDS=${POLL_SECONDS:-60}
MAX_WAIT_SECONDS=${MAX_WAIT_SECONDS:-1209600}
EXPECTED_HOSTNAME=${EXPECTED_HOSTNAME:-user-Super-Server}

PLAN="$TOOL_REPO/IN_PLACE_TTT_BASELINE_QWEN3_4B_BASE_SCALE_FEASIBILITY_PLAN.md"
REPORTER="$TOOL_REPO/scripts/summarize_in_place_ttt_qwen3_4b_base_scale_feasibility.py"
AUDITOR="$TOOL_REPO/scripts/audit_in_place_ttt_qwen3_4b_base_scale_feasibility.py"
STAGE1_AUDIT="$STAGE1_WATCH_ROOT/corrected_independent_audit.json"
STAGE2_AUDIT="$STAGE2_WATCH_ROOT/corrected_independent_audit.json"
SUMMARY_JSON="$OUTPUT_ROOT/scale_feasibility.json"
REPORT_MD="$OUTPUT_ROOT/SCALE_FEASIBILITY.md"

if [[ "$(hostname)" != "$EXPECTED_HOSTNAME" ]]; then
  echo "scale-feasibility reporter is restricted to hku-gpu2" >&2; exit 2
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "refusing to reuse scale-feasibility output root" >&2; exit 3
fi
if [[ "$POLL_SECONDS" -lt 10 || "$MAX_WAIT_SECONDS" -lt "$POLL_SECONDS" ]]; then
  echo "invalid wait interval or budget" >&2; exit 2
fi
test "$(git -C "$TOOL_REPO" rev-parse HEAD)" = "$TOOL_COMMIT"
test -z "$(git -C "$TOOL_REPO" status --short)"
for path in "$PYTHON_BIN" "$PLAN" "$REPORTER" "$AUDITOR"; do test -s "$path"; done

mkdir -p "$OUTPUT_ROOT"
sha256sum "$0" "$PLAN" "$REPORTER" "$AUDITOR" \
  > "$OUTPUT_ROOT/frozen_inputs.sha256"
write_state() {
  printf '%s\n' "$1" > "$OUTPUT_ROOT/state.txt.partial"
  mv "$OUTPUT_ROOT/state.txt.partial" "$OUTPUT_ROOT/state.txt"
  date --iso-8601=seconds > "$OUTPUT_ROOT/state_time.txt.partial"
  mv "$OUTPUT_ROOT/state_time.txt.partial" "$OUTPUT_ROOT/state_time.txt"
}

started=$(date +%s)
while true; do
  write_state waiting_for_corrected_stage1_and_stage2_timing
  stage1_state=$(cat "$STAGE1_WATCH_ROOT/state.txt" 2>/dev/null || true)
  stage2_state=$(cat "$STAGE2_WATCH_ROOT/state.txt" 2>/dev/null || true)
  printf '%s\n' "$stage1_state" > "$OUTPUT_ROOT/last_stage1_state.txt"
  printf '%s\n' "$stage2_state" > "$OUTPUT_ROOT/last_stage2_state.txt"
  tmux list-panes -t "$STAGE1_SESSION" \
    -F '#{pane_pid} #{pane_dead} #{pane_current_command}' \
    > "$OUTPUT_ROOT/last_stage1_handle.txt" 2>&1 || true
  tmux list-panes -t "$STAGE2_SESSION" \
    -F '#{pane_pid} #{pane_dead} #{pane_current_command}' \
    > "$OUTPUT_ROOT/last_stage2_handle.txt" 2>&1 || true
  stage1_ready=0; stage2_ready=0
  case "$stage1_state" in
    completed_corrected_stage1_timing) stage1_ready=1 ;;
    failed_*|invalid_*|timed_out_*|upstream_*|*_handle_missing)
      write_state "upstream_${stage1_state}"; exit 4 ;;
    *) tmux has-session -t "$STAGE1_SESSION" 2>/dev/null || {
         write_state corrected_stage1_timing_handle_missing; exit 5; } ;;
  esac
  case "$stage2_state" in
    completed_corrected_stage2_timing) stage2_ready=1 ;;
    failed_*|invalid_*|timed_out_*|upstream_*|*_handle_missing)
      write_state "upstream_${stage2_state}"; exit 6 ;;
    *) tmux has-session -t "$STAGE2_SESSION" 2>/dev/null || {
         write_state corrected_stage2_timing_handle_missing; exit 7; } ;;
  esac
  [[ "$stage1_ready" -eq 0 || "$stage2_ready" -eq 0 ]] || break
  now=$(date +%s)
  [[ $((now - started)) -lt "$MAX_WAIT_SECONDS" ]] || {
    write_state timed_out_waiting_for_corrected_timing; exit 8;
  }
  sleep "$POLL_SECONDS"
done

test -s "$STAGE1_AUDIT"; test -s "$STAGE2_AUDIT"
sha256sum "$STAGE1_AUDIT" "$STAGE2_AUDIT" \
  > "$OUTPUT_ROOT/released_inputs.sha256"
write_state building_scale_feasibility_report
env CUDA_VISIBLE_DEVICES= "$PYTHON_BIN" "$REPORTER" \
  --stage1-audit "$STAGE1_AUDIT" --stage2-audit "$STAGE2_AUDIT" \
  --plan "$PLAN" --output-json "$SUMMARY_JSON" --output-md "$REPORT_MD" \
  > "$OUTPUT_ROOT/reporter.stdout"
write_state auditing_scale_feasibility_report
env CUDA_VISIBLE_DEVICES= "$PYTHON_BIN" "$AUDITOR" \
  --stage1-audit "$STAGE1_AUDIT" --stage2-audit "$STAGE2_AUDIT" \
  --plan "$PLAN" --reporter "$REPORTER" --summary-json "$SUMMARY_JSON" \
  --report-md "$REPORT_MD" --output "$OUTPUT_ROOT/independent_audit.json" \
  > "$OUTPUT_ROOT/independent_audit.stdout"
write_state completed_scale_feasibility_report
