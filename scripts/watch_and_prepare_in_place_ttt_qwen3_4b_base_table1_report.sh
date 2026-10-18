#!/usr/bin/env bash
set -euo pipefail

TOOL_REPO=${TOOL_REPO:?set the clean CBF-TTT checkout}
TOOL_COMMIT=${TOOL_COMMIT:?set the exact tool commit}
PYTHON_BIN=${PYTHON_BIN:?set the isolated OpenCompass Python}
MATRIX_ROOT=${MATRIX_ROOT:?set the corrected primary matrix root}
MATRIX_WATCH_ROOT=${MATRIX_WATCH_ROOT:?set its watcher root}
MATRIX_SESSION=${MATRIX_SESSION:?set its tmux session}
FORMAL_256K_ROOT=${FORMAL_256K_ROOT:?set the corrected formal 256K root}
FORMAL_256K_WATCH_ROOT=${FORMAL_256K_WATCH_ROOT:?set its watcher root}
FORMAL_256K_SESSION=${FORMAL_256K_SESSION:?set its tmux session}
CORRECTED_PRIMARY_RESOURCE_AUDIT=${CORRECTED_PRIMARY_RESOURCE_AUDIT:?set the corrected primary resource audit}
CORRECTED_256K_RESOURCE_AUDIT=${CORRECTED_256K_RESOURCE_AUDIT:?set the corrected 256K resource audit}
OUTPUT_ROOT=${OUTPUT_ROOT:?set a new final Table 1 report root}
POLL_SECONDS=${POLL_SECONDS:-60}
MAX_WAIT_SECONDS=${MAX_WAIT_SECONDS:-1209600}
EXPECTED_HOSTNAME=${EXPECTED_HOSTNAME:-user-Super-Server}

PLAN="$TOOL_REPO/IN_PLACE_TTT_BASELINE_QWEN3_4B_BASE_TABLE1_REPORT_PLAN.md"
REPORTER="$TOOL_REPO/scripts/summarize_in_place_ttt_qwen3_4b_base_table1.py"
AUDITOR="$TOOL_REPO/scripts/audit_in_place_ttt_qwen3_4b_base_table1.py"
CORRECTED_MATRIX_AUDIT="$MATRIX_WATCH_ROOT/corrected_independent_audit.json"
CORRECTED_FORMAL_256K_AUDIT="$FORMAL_256K_WATCH_ROOT/corrected_independent_audit.json"
SUMMARY_JSON="$OUTPUT_ROOT/table1_summary.json"
REPORT_MD="$OUTPUT_ROOT/TABLE1_RESULT.md"

if [[ "$(hostname)" != "$EXPECTED_HOSTNAME" ]]; then
  echo "final Table 1 reporter is restricted to hku-gpu2" >&2; exit 2
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "refusing to reuse final Table 1 report root" >&2; exit 3
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
  write_state waiting_for_corrected_table1_inputs
  matrix_state=$(cat "$MATRIX_WATCH_ROOT/state.txt" 2>/dev/null || true)
  formal_state=$(cat "$FORMAL_256K_WATCH_ROOT/state.txt" 2>/dev/null || true)
  printf '%s\n' "$matrix_state" > "$OUTPUT_ROOT/last_matrix_state.txt"
  printf '%s\n' "$formal_state" > "$OUTPUT_ROOT/last_formal_256k_state.txt"
  tmux list-panes -t "$MATRIX_SESSION" \
    -F '#{pane_pid} #{pane_dead} #{pane_current_command}' \
    > "$OUTPUT_ROOT/last_matrix_handle.txt" 2>&1 || true
  tmux list-panes -t "$FORMAL_256K_SESSION" \
    -F '#{pane_pid} #{pane_dead} #{pane_current_command}' \
    > "$OUTPUT_ROOT/last_formal_256k_handle.txt" 2>&1 || true
  matrix_ready=0; formal_ready=0
  case "$matrix_state" in
    completed_corrected_ruler_matrix) matrix_ready=1 ;;
    failed_*|invalid_*|timed_out_*|upstream_*|*_handle_missing)
      write_state "upstream_${matrix_state}"; exit 4 ;;
    *) tmux has-session -t "$MATRIX_SESSION" 2>/dev/null || {
         write_state corrected_matrix_handle_missing; exit 5; } ;;
  esac
  case "$formal_state" in
    completed_corrected_formal_ruler_256k) formal_ready=1 ;;
    failed_*|invalid_*|timed_out_*|upstream_*|*_handle_missing)
      write_state "upstream_${formal_state}"; exit 6 ;;
    *) tmux has-session -t "$FORMAL_256K_SESSION" 2>/dev/null || {
         write_state corrected_formal_256k_handle_missing; exit 7; } ;;
  esac
  [[ "$matrix_ready" -eq 0 || "$formal_ready" -eq 0 ]] || break
  now=$(date +%s)
  [[ $((now - started)) -lt "$MAX_WAIT_SECONDS" ]] || {
    write_state timed_out_waiting_for_corrected_table1_inputs; exit 8;
  }
  sleep "$POLL_SECONDS"
done

for path in "$CORRECTED_MATRIX_AUDIT" "$CORRECTED_FORMAL_256K_AUDIT" \
  "$CORRECTED_PRIMARY_RESOURCE_AUDIT" "$CORRECTED_256K_RESOURCE_AUDIT"; do
  test -s "$path"
done
sha256sum "$CORRECTED_MATRIX_AUDIT" "$CORRECTED_FORMAL_256K_AUDIT" \
  "$CORRECTED_PRIMARY_RESOURCE_AUDIT" "$CORRECTED_256K_RESOURCE_AUDIT" \
  > "$OUTPUT_ROOT/released_inputs.sha256"
write_state building_corrected_table1_report
env CUDA_VISIBLE_DEVICES= "$PYTHON_BIN" "$REPORTER" \
  --matrix-root "$MATRIX_ROOT" --corrected-matrix-audit "$CORRECTED_MATRIX_AUDIT" \
  --formal-256k-root "$FORMAL_256K_ROOT" \
  --corrected-formal-256k-audit "$CORRECTED_FORMAL_256K_AUDIT" \
  --corrected-primary-resource-audit "$CORRECTED_PRIMARY_RESOURCE_AUDIT" \
  --corrected-256k-resource-audit "$CORRECTED_256K_RESOURCE_AUDIT" \
  --plan "$PLAN" --output-json "$SUMMARY_JSON" --output-md "$REPORT_MD" \
  > "$OUTPUT_ROOT/reporter.stdout"
write_state auditing_corrected_table1_report
env CUDA_VISIBLE_DEVICES= "$PYTHON_BIN" "$AUDITOR" \
  --matrix-root "$MATRIX_ROOT" --corrected-matrix-audit "$CORRECTED_MATRIX_AUDIT" \
  --formal-256k-root "$FORMAL_256K_ROOT" \
  --corrected-formal-256k-audit "$CORRECTED_FORMAL_256K_AUDIT" \
  --corrected-primary-resource-audit "$CORRECTED_PRIMARY_RESOURCE_AUDIT" \
  --corrected-256k-resource-audit "$CORRECTED_256K_RESOURCE_AUDIT" \
  --plan "$PLAN" --reporter "$REPORTER" --summary-json "$SUMMARY_JSON" \
  --report-md "$REPORT_MD" --output "$OUTPUT_ROOT/independent_audit.json" \
  > "$OUTPUT_ROOT/independent_audit.stdout"
write_state completed_corrected_table1_report
