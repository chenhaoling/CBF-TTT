#!/usr/bin/env bash
set -euo pipefail

UPSTREAM_REPO=${UPSTREAM_REPO:?set the clean official checkout}
TOOL_REPO=${TOOL_REPO:?set the clean CBF-TTT checkout}
TOOL_COMMIT=${TOOL_COMMIT:?set the exact tool commit}
PYTHON_BIN=${PYTHON_BIN:?set the isolated OpenCompass Python}
TOKENIZER=${TOKENIZER:?set the original Qwen3-4B tokenizer directory}
TRAINING_ROOT=${TRAINING_ROOT:?set the future paired formal Stage-2 root}
TRAINING_SESSION=${TRAINING_SESSION:?set the formal Stage-2 watcher tmux session}
OUTPUT_ROOT=${OUTPUT_ROOT:?set a new RULER resource-ladder root}
WATCH_ROOT=${WATCH_ROOT:-${OUTPUT_ROOT}.watch}
PUBLIC_CORPUS_PLAN=${PUBLIC_CORPUS_PLAN:-$TOOL_REPO/IN_PLACE_TTT_BASELINE_PUBLIC_CORPUS_PLAN.md}
EVALUATION_PLAN=${EVALUATION_PLAN:-$TOOL_REPO/IN_PLACE_TTT_BASELINE_2X5090_RULER_PLAN.md}
POLL_SECONDS=${POLL_SECONDS:-60}
MAX_WAIT_SECONDS=${MAX_WAIT_SECONDS:-604800}
EXPECTED_HOSTNAME=${EXPECTED_HOSTNAME:-user-Super-Server}
UPSTREAM_COMMIT=be2324829b0e91c8fd10a74d4b43714fde6676e1

if [[ "$(hostname)" != "$EXPECTED_HOSTNAME" ]]; then
  echo "the RULER resource watcher is restricted to hku-gpu2" >&2; exit 2
fi
if [[ -e "$OUTPUT_ROOT" || -e "${OUTPUT_ROOT}.partial" || -e "$WATCH_ROOT" ]]; then
  echo "refusing to reuse RULER resource or watcher output" >&2; exit 3
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

mkdir -p "$WATCH_ROOT"
sha256sum \
  "$0" "$TOOL_REPO/scripts/run_in_place_ttt_2x5090_ruler_resource_ladder.sh" \
  "$TOOL_REPO/scripts/run_in_place_ttt_2x5090_ruler_single.sh" \
  "$TOOL_REPO/scripts/audit_in_place_ttt_2x5090_ruler_resource_ladder.py" \
  "$PUBLIC_CORPUS_PLAN" "$EVALUATION_PLAN" "$TOKENIZER/tokenizer.json" \
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
  for source in "$TRAINING_ROOT/state.txt" "${TRAINING_ROOT}.watch/state.txt"; do
    name=$(basename "$(dirname "$source")")_$(basename "$source")
    if [[ -f "$source" ]]; then cp "$source" "$WATCH_ROOT/${prefix}_${name}"
    else : > "$WATCH_ROOT/${prefix}_${name}"; fi
  done
  tmux list-panes -t "$TRAINING_SESSION" \
    -F '#{pane_pid} #{pane_dead} #{pane_current_command}' \
    > "$WATCH_ROOT/${prefix}_training_handle.txt" 2>&1 || true
}

training_valid() {
  "$PYTHON_BIN" - "$TRAINING_ROOT/independent_audit.json" <<'PY'
import json, pathlib, sys
audit = json.loads(pathlib.Path(sys.argv[1]).read_text())
if (audit.get("protocol") != "in_place_ttt_2x5090_100m_stage2_audit_v1"
        or audit.get("passed") is not True or audit.get("errors") != []
        or set(audit.get("arms", {})) != {"no_ttt", "ttt"}):
    raise SystemExit(1)
PY
}

started=$(date +%s)
while true; do
  write_state waiting_for_audited_formal_stage2
  snapshot last
  training_state=$(cat "$TRAINING_ROOT/state.txt" 2>/dev/null || true)
  if [[ "$training_state" == completed_100m_stage2 ]]; then
    if ! training_valid; then write_state invalid_formal_stage2; exit 4; fi
    break
  fi
  upstream_state=$(cat "${TRAINING_ROOT}.watch/state.txt" 2>/dev/null || true)
  case "$upstream_state" in
    failed_*|invalid_*|timed_out_*|upstream_*|stage2_*_missing)
      write_state "upstream_${upstream_state}"; exit 5 ;;
  esac
  if ! tmux has-session -t "$TRAINING_SESSION" 2>/dev/null; then
    snapshot missing_handle
    write_state formal_stage2_watcher_handle_missing
    exit 6
  fi
  now=$(date +%s)
  if [[ $((now - started)) -ge "$MAX_WAIT_SECONDS" ]]; then
    write_state timed_out_waiting_for_formal_stage2; exit 7
  fi
  sleep "$POLL_SECONDS"
done

write_state launching_ruler_resource_ladder
set +e
env UPSTREAM_REPO="$UPSTREAM_REPO" TOOL_REPO="$TOOL_REPO" \
  TOOL_COMMIT="$TOOL_COMMIT" PYTHON_BIN="$PYTHON_BIN" TOKENIZER="$TOKENIZER" \
  TRAINING_ROOT="$TRAINING_ROOT" OUTPUT_ROOT="$OUTPUT_ROOT" \
  PUBLIC_CORPUS_PLAN="$PUBLIC_CORPUS_PLAN" EVALUATION_PLAN="$EVALUATION_PLAN" \
  EXPECTED_HOSTNAME="$EXPECTED_HOSTNAME" \
  bash "$TOOL_REPO/scripts/run_in_place_ttt_2x5090_ruler_resource_ladder.sh"
status=$?
set -e
printf '%s\n' "$status" > "$WATCH_ROOT/resource_runner_exit_code.txt"
if [[ "$status" -ne 0 ]]; then write_state failed_ruler_resource_ladder; exit 8; fi
if [[ "$(cat "$OUTPUT_ROOT/state.txt" 2>/dev/null || true)" \
      != completed_ruler_resource_ladder \
      || ! -s "$OUTPUT_ROOT/independent_resource_ladder_audit.json" ]]; then
  write_state invalid_ruler_resource_ladder_output; exit 9
fi
if ! "$PYTHON_BIN" - "$OUTPUT_ROOT/independent_resource_ladder_audit.json" <<'PY'
import json, pathlib, sys
audit = json.loads(pathlib.Path(sys.argv[1]).read_text())
expected = {f"{suffix}_{arm}" for suffix in ("4k", "32k", "64k", "128k")
            for arm in ("no_ttt", "ttt")}
if (audit.get("protocol")
        != "in_place_ttt_2x5090_ruler_resource_ladder_audit_v1"
        or audit.get("passed") is not True or audit.get("errors") != []
        or audit.get("scores_discarded") is not True
        or set(audit.get("boundaries", {})) != expected):
    raise SystemExit(1)
PY
then
  write_state invalid_ruler_resource_ladder_audit; exit 10
fi
write_state completed_ruler_resource_ladder
