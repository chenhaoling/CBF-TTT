#!/usr/bin/env bash
set -euo pipefail

UPSTREAM_REPO=${UPSTREAM_REPO:?set the clean official checkout}
TOOL_REPO=${TOOL_REPO:?set the clean CBF-TTT checkout}
TOOL_COMMIT=${TOOL_COMMIT:?set the exact tool commit}
PYTHON_BIN=${PYTHON_BIN:?set the isolated OpenCompass Python}
TOKENIZER=${TOKENIZER:?set the original Qwen3-4B tokenizer directory}
TRAINING_ROOT=${TRAINING_ROOT:?set the completed paired formal Stage-2 root}
MATRIX_ROOT=${MATRIX_ROOT:?set the future completed 4K-128K matrix root}
MATRIX_SESSION=${MATRIX_SESSION:?set the formal matrix watcher tmux session}
OUTPUT_ROOT=${OUTPUT_ROOT:?set a new paired 256K resource root}
WATCH_ROOT=${WATCH_ROOT:-${OUTPUT_ROOT}.watch}
PUBLIC_CORPUS_PLAN=${PUBLIC_CORPUS_PLAN:-$TOOL_REPO/IN_PLACE_TTT_BASELINE_PUBLIC_CORPUS_PLAN.md}
EVALUATION_PLAN=${EVALUATION_PLAN:-$TOOL_REPO/IN_PLACE_TTT_BASELINE_2X5090_RULER_PLAN.md}
EXTRAPOLATION_PLAN=${EXTRAPOLATION_PLAN:-$TOOL_REPO/IN_PLACE_TTT_BASELINE_2X5090_RULER_256K_PLAN.md}
POLL_SECONDS=${POLL_SECONDS:-60}
MAX_WAIT_SECONDS=${MAX_WAIT_SECONDS:-604800}
EXPECTED_HOSTNAME=${EXPECTED_HOSTNAME:-user-Super-Server}
UPSTREAM_COMMIT=be2324829b0e91c8fd10a74d4b43714fde6676e1

if [[ "$(hostname)" != "$EXPECTED_HOSTNAME" ]]; then
  echo "the RULER-256K resource watcher is restricted to hku-gpu2" >&2; exit 2
fi
if [[ -e "$OUTPUT_ROOT" || -e "${OUTPUT_ROOT}.partial" || -e "$WATCH_ROOT" ]]; then
  echo "refusing to reuse 256K resource or watcher output" >&2; exit 3
fi
if [[ "$POLL_SECONDS" -lt 10 || "$MAX_WAIT_SECONDS" -lt "$POLL_SECONDS" ]]; then
  echo "invalid watcher interval/budget" >&2; exit 2
fi
test "$(git -C "$UPSTREAM_REPO" rev-parse HEAD)" = "$UPSTREAM_COMMIT"
test -z "$(git -C "$UPSTREAM_REPO" status --short)"
test "$(git -C "$TOOL_REPO" rev-parse HEAD)" = "$TOOL_COMMIT"
test -z "$(git -C "$TOOL_REPO" status --short)"
test -x "$PYTHON_BIN"; test -s "$TOKENIZER/tokenizer.json"
test -s "$PUBLIC_CORPUS_PLAN"; test -s "$EVALUATION_PLAN"
test -s "$EXTRAPOLATION_PLAN"

mkdir -p "$WATCH_ROOT"
sha256sum "$0" \
  "$TOOL_REPO/scripts/run_in_place_ttt_2x5090_ruler_256k_resource.sh" \
  "$TOOL_REPO/scripts/audit_in_place_ttt_2x5090_ruler_256k_resource.py" \
  "$PUBLIC_CORPUS_PLAN" "$EVALUATION_PLAN" "$EXTRAPOLATION_PLAN" \
  "$TOKENIZER/tokenizer.json" > "$WATCH_ROOT/frozen_inputs.sha256"

write_state() {
  printf '%s\n' "$1" > "$WATCH_ROOT/state.txt.partial"
  mv "$WATCH_ROOT/state.txt.partial" "$WATCH_ROOT/state.txt"
  date --iso-8601=seconds > "$WATCH_ROOT/state_time.txt.partial"
  mv "$WATCH_ROOT/state_time.txt.partial" "$WATCH_ROOT/state_time.txt"
}
snapshot() {
  local prefix=$1 source name
  date --iso-8601=seconds > "$WATCH_ROOT/${prefix}_time.txt"
  for source in "$MATRIX_ROOT/state.txt" "${MATRIX_ROOT}.watch/state.txt"; do
    name=$(basename "$(dirname "$source")")_$(basename "$source")
    if [[ -f "$source" ]]; then cp "$source" "$WATCH_ROOT/${prefix}_${name}"
    else : > "$WATCH_ROOT/${prefix}_${name}"; fi
  done
  tmux list-panes -t "$MATRIX_SESSION" \
    -F '#{pane_pid} #{pane_dead} #{pane_current_command}' \
    > "$WATCH_ROOT/${prefix}_matrix_handle.txt" 2>&1 || true
}
matrix_valid() {
  "$PYTHON_BIN" - "$MATRIX_ROOT/independent_matrix_audit.json" <<'PY'
import json, pathlib, sys
audit = json.loads(pathlib.Path(sys.argv[1]).read_text())
if (audit.get("protocol") != "in_place_ttt_2x5090_ruler_matrix_audit_v1"
        or audit.get("passed") is not True or audit.get("errors") != []
        or set(audit.get("recomputed_context_scores", {}))
        != {"4k", "8k", "16k", "32k", "64k", "128k"}):
    raise SystemExit(1)
PY
}

started=$(date +%s)
while true; do
  write_state waiting_for_audited_ruler_matrix
  snapshot last
  matrix_state=$(cat "$MATRIX_ROOT/state.txt" 2>/dev/null || true)
  if [[ "$matrix_state" == completed_ruler_matrix ]]; then
    if ! matrix_valid; then write_state invalid_ruler_matrix; exit 4; fi
    break
  fi
  upstream_state=$(cat "${MATRIX_ROOT}.watch/state.txt" 2>/dev/null || true)
  case "$upstream_state" in
    failed_*|invalid_*|timed_out_*|upstream_*|ruler_*_missing)
      write_state "upstream_${upstream_state}"; exit 5 ;;
  esac
  if ! tmux has-session -t "$MATRIX_SESSION" 2>/dev/null; then
    snapshot missing_handle
    write_state ruler_matrix_watcher_handle_missing; exit 6
  fi
  now=$(date +%s)
  if [[ $((now - started)) -ge "$MAX_WAIT_SECONDS" ]]; then
    write_state timed_out_waiting_for_ruler_matrix; exit 7
  fi
  sleep "$POLL_SECONDS"
done

write_state launching_ruler_256k_resource
set +e
env UPSTREAM_REPO="$UPSTREAM_REPO" TOOL_REPO="$TOOL_REPO" \
  TOOL_COMMIT="$TOOL_COMMIT" PYTHON_BIN="$PYTHON_BIN" TOKENIZER="$TOKENIZER" \
  TRAINING_ROOT="$TRAINING_ROOT" MATRIX_ROOT="$MATRIX_ROOT" \
  OUTPUT_ROOT="$OUTPUT_ROOT" PUBLIC_CORPUS_PLAN="$PUBLIC_CORPUS_PLAN" \
  EVALUATION_PLAN="$EVALUATION_PLAN" EXTRAPOLATION_PLAN="$EXTRAPOLATION_PLAN" \
  EXPECTED_HOSTNAME="$EXPECTED_HOSTNAME" \
  bash "$TOOL_REPO/scripts/run_in_place_ttt_2x5090_ruler_256k_resource.sh"
status=$?
set -e
printf '%s\n' "$status" > "$WATCH_ROOT/resource_runner_exit_code.txt"
if [[ "$status" -ne 0 ]]; then write_state failed_ruler_256k_resource; exit 8; fi
if [[ "$(cat "$OUTPUT_ROOT/state.txt" 2>/dev/null || true)" \
      != completed_ruler_256k_resource \
      || ! -s "$OUTPUT_ROOT/independent_resource_audit.json" ]]; then
  write_state invalid_ruler_256k_resource_output; exit 9
fi
if ! "$PYTHON_BIN" - "$OUTPUT_ROOT/independent_resource_audit.json" <<'PY'
import json, pathlib, sys
audit = json.loads(pathlib.Path(sys.argv[1]).read_text())
if (audit.get("protocol")
        != "in_place_ttt_2x5090_ruler_256k_resource_audit_v1"
        or audit.get("passed") is not True or audit.get("errors") != []
        or audit.get("formal_256k_released") is not True
        or set(audit.get("boundaries", {})) != {"no_ttt", "ttt"}):
    raise SystemExit(1)
PY
then
  write_state invalid_ruler_256k_resource_audit; exit 10
fi
write_state completed_ruler_256k_resource
