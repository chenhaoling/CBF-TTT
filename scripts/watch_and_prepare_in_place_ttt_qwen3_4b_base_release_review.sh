#!/usr/bin/env bash
set -euo pipefail

STAGE=${STAGE:?set stage1 or stage2}
TOOL_REPO=${TOOL_REPO:?set the clean CBF-TTT checkout}
TOOL_COMMIT=${TOOL_COMMIT:?set the exact tool commit}
PYTHON_BIN=${PYTHON_BIN:?set the official training Python}
TIMING_ROOT=${TIMING_ROOT:?set the corrected timing result root}
TIMING_WATCH_ROOT=${TIMING_WATCH_ROOT:?set the corrected timing watcher root}
TIMING_SESSION=${TIMING_SESSION:?set the corrected timing tmux session}
SELECTED_STREAM_ROOT=${SELECTED_STREAM_ROOT:?set the corrected selected-stream root}
SELECTED_STREAM_SESSION=${SELECTED_STREAM_SESSION:?set the selected-stream tmux session}
TARGET_RELEASE=${TARGET_RELEASE:?set the intended future release path}
OUTPUT_ROOT=${OUTPUT_ROOT:?set a new review output root}
POLL_SECONDS=${POLL_SECONDS:-60}
MAX_WAIT_SECONDS=${MAX_WAIT_SECONDS:-1209600}
EXPECTED_HOSTNAME=${EXPECTED_HOSTNAME:-user-Super-Server}

PLAN="$TOOL_REPO/IN_PLACE_TTT_BASELINE_QWEN3_4B_BASE_RELEASE_REVIEW_ADDENDUM.md"
BUILDER="$TOOL_REPO/scripts/create_in_place_ttt_qwen3_4b_base_release_review.py"
SELECTED_AUDIT="$SELECTED_STREAM_ROOT/independent_audit.json"
CORRECTED_TIMING_AUDIT="$TIMING_WATCH_ROOT/corrected_independent_audit.json"
TIMING_AUDIT="$TIMING_ROOT/independent_audit.json"
REVIEW="$OUTPUT_ROOT/review.json"

case "$STAGE" in
  stage1)
    TIMING_DONE=completed_corrected_stage1_timing
    RELEASE_TOOL="$TOOL_REPO/scripts/create_in_place_ttt_qwen3_4b_base_stage1_release.py"
    ;;
  stage2)
    TIMING_DONE=completed_corrected_stage2_timing
    RELEASE_TOOL="$TOOL_REPO/scripts/create_in_place_ttt_qwen3_4b_base_stage2_release.py"
    ;;
  *) echo "STAGE must be stage1 or stage2" >&2; exit 2 ;;
esac

if [[ "$(hostname)" != "$EXPECTED_HOSTNAME" ]]; then
  echo "formal-release review is restricted to hku-gpu2" >&2; exit 2
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "refusing to reuse review root $OUTPUT_ROOT" >&2; exit 3
fi
if [[ -e "$TARGET_RELEASE" || -e "${TARGET_RELEASE}.partial" ]]; then
  echo "target release already exists before review" >&2; exit 3
fi
if [[ "$POLL_SECONDS" -lt 10 || "$MAX_WAIT_SECONDS" -lt "$POLL_SECONDS" ]]; then
  echo "invalid wait interval or budget" >&2; exit 2
fi
test "$(git -C "$TOOL_REPO" rev-parse HEAD)" = "$TOOL_COMMIT"
test -z "$(git -C "$TOOL_REPO" status --short)"
for path in "$PYTHON_BIN" "$PLAN" "$BUILDER" "$RELEASE_TOOL"; do test -s "$path"; done

mkdir -p "$OUTPUT_ROOT"
sha256sum "$0" "$PLAN" "$BUILDER" "$RELEASE_TOOL" \
  > "$OUTPUT_ROOT/frozen_inputs.sha256"
printf '%s\n' "$TOOL_COMMIT" > "$OUTPUT_ROOT/tool_commit.txt"

write_state() {
  printf '%s\n' "$1" > "$OUTPUT_ROOT/state.txt.partial"
  mv "$OUTPUT_ROOT/state.txt.partial" "$OUTPUT_ROOT/state.txt"
  date --iso-8601=seconds > "$OUTPUT_ROOT/state_time.txt.partial"
  mv "$OUTPUT_ROOT/state_time.txt.partial" "$OUTPUT_ROOT/state_time.txt"
}

on_error() {
  local code=$?
  trap - ERR
  printf 'exit=%s state=%s\n' "$code" "$(cat "$OUTPUT_ROOT/state.txt" 2>/dev/null || true)" \
    >> "$OUTPUT_ROOT/error.log"
  write_state "failed_${STAGE}_release_review"
  exit "$code"
}
trap on_error ERR

started=$(date +%s)
while true; do
  write_state "waiting_for_${STAGE}_timing_and_selected_stream"
  timing_state=$(cat "$TIMING_WATCH_ROOT/state.txt" 2>/dev/null || true)
  selected_state=$(cat "$SELECTED_STREAM_ROOT/state.txt" 2>/dev/null || true)
  printf '%s\n' "$timing_state" > "$OUTPUT_ROOT/last_timing_state.txt"
  printf '%s\n' "$selected_state" > "$OUTPUT_ROOT/last_selected_stream_state.txt"
  tmux list-panes -t "$TIMING_SESSION" -F '#{pane_pid} #{pane_dead} #{pane_current_command}' \
    > "$OUTPUT_ROOT/last_timing_handle.txt" 2>&1 || true
  tmux list-panes -t "$SELECTED_STREAM_SESSION" \
    -F '#{pane_pid} #{pane_dead} #{pane_current_command}' \
    > "$OUTPUT_ROOT/last_selected_stream_handle.txt" 2>&1 || true

  if [[ -e "$TARGET_RELEASE" || -e "${TARGET_RELEASE}.partial" ]]; then
    echo "target release appeared before review packet" >&2; false
  fi
  case "$timing_state" in
    "$TIMING_DONE") timing_ready=1 ;;
    failed_*|invalid_*|timed_out_*|upstream_*|*_handle_missing)
      echo "timing prerequisite failed: $timing_state" >&2; false ;;
    *)
      timing_ready=0
      tmux has-session -t "$TIMING_SESSION" 2>/dev/null || {
        echo "timing handle missing in state $timing_state" >&2; false;
      }
      ;;
  esac
  case "$selected_state" in
    completed_selected_stream_audit) selected_ready=1 ;;
    failed_*|invalid_*|timed_out_*|upstream_*|*_handle_missing)
      echo "selected-stream prerequisite failed: $selected_state" >&2; false ;;
    *)
      selected_ready=0
      tmux has-session -t "$SELECTED_STREAM_SESSION" 2>/dev/null || {
        echo "selected-stream handle missing in state $selected_state" >&2; false;
      }
      ;;
  esac
  if [[ "$timing_ready" -eq 1 && "$selected_ready" -eq 1 ]]; then
    break
  fi
  now=$(date +%s)
  if [[ $((now - started)) -ge "$MAX_WAIT_SECONDS" ]]; then
    echo "timed out waiting for review prerequisites" >&2; false
  fi
  sleep "$POLL_SECONDS"
done

write_state "building_${STAGE}_release_review"
test -s "$TIMING_AUDIT"
test -s "$CORRECTED_TIMING_AUDIT"
test -s "$SELECTED_AUDIT"
env CUDA_VISIBLE_DEVICES= "$PYTHON_BIN" "$BUILDER" \
  --stage "$STAGE" --timing-audit "$TIMING_AUDIT" \
  --corrected-timing-audit "$CORRECTED_TIMING_AUDIT" \
  --selected-stream-audit "$SELECTED_AUDIT" --release-tool "$RELEASE_TOOL" \
  --plan "$PLAN" --target-release "$TARGET_RELEASE" --output "$REVIEW" \
  --expected-hostname "$EXPECTED_HOSTNAME" > "$OUTPUT_ROOT/review.stdout"

env CUDA_VISIBLE_DEVICES= "$PYTHON_BIN" - "$REVIEW" "$TIMING_AUDIT" \
  "$CORRECTED_TIMING_AUDIT" "$SELECTED_AUDIT" "$TARGET_RELEASE" <<'PY'
import hashlib, json, pathlib, sys
review_path, timing, corrected, selected, target = map(pathlib.Path, sys.argv[1:])
payload = json.loads(review_path.read_text())
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
assert payload["protocol"] == "in_place_ttt_qwen3_4b_base_100m_release_review_v1"
assert payload["ready_for_human_review"] is True
assert payload["formal_release_created"] is False
assert payload["target_release_path"] == str(target.resolve())
assert not target.exists() and not target.with_name(target.name + ".partial").exists()
assert payload["bindings"]["timing_audit_sha256"] == sha(timing)
assert payload["bindings"]["corrected_timing_audit_sha256"] == sha(corrected)
assert payload["bindings"]["selected_stream_audit_sha256"] == sha(selected)
assert set(payload["required_human_fields"]) == {"accept", "reviewed_by", "decision_note"}
PY
sha256sum "$TIMING_AUDIT" "$CORRECTED_TIMING_AUDIT" "$SELECTED_AUDIT" \
  "$REVIEW" > "$OUTPUT_ROOT/final_outputs.sha256"
write_state "completed_${STAGE}_release_review"
trap - ERR
