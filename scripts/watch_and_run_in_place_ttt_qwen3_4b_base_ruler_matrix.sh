#!/usr/bin/env bash
set -euo pipefail

UPSTREAM_REPO=${UPSTREAM_REPO:?set the clean official checkout}
TOOL_REPO=${TOOL_REPO:?set the clean CBF-TTT checkout}
TOOL_COMMIT=${TOOL_COMMIT:?set the exact tool commit}
PYTHON_BIN=${PYTHON_BIN:?set the isolated OpenCompass Python}
TOKENIZER=${TOKENIZER:?set the exact Qwen3-4B-Base tokenizer directory}
MODEL_IDENTITY=${MODEL_IDENTITY:?set the passing Base identity JSON}
TRAINING_ROOT=${TRAINING_ROOT:?set the corrected formal Stage-2 root}
CORRECTED_TRAINING_AUDIT=${CORRECTED_TRAINING_AUDIT:?set the corrected training audit}
RESOURCE_ROOT=${RESOURCE_ROOT:?set the corrected RULER resource root}
RESOURCE_WATCH_ROOT=${RESOURCE_WATCH_ROOT:?set the corrected resource watcher root}
RESOURCE_SESSION=${RESOURCE_SESSION:?set the corrected resource tmux session}
OUTPUT_ROOT=${OUTPUT_ROOT:?set a new corrected RULER matrix root}
WATCH_ROOT=${WATCH_ROOT:-${OUTPUT_ROOT}.watch}
PUBLIC_CORPUS_PLAN=${PUBLIC_CORPUS_PLAN:?set the public-corpus plan}
EVALUATION_PLAN=${EVALUATION_PLAN:?set the frozen 2x5090 RULER plan}
CORRECTED_PLAN=${CORRECTED_PLAN:?set the corrected Base RULER plan}
PAPER_INFERENCE_ADDENDUM=${PAPER_INFERENCE_ADDENDUM:?set the paper inference addendum}
POLL_SECONDS=${POLL_SECONDS:-60}
MAX_WAIT_SECONDS=${MAX_WAIT_SECONDS:-1209600}
EXPECTED_HOSTNAME=${EXPECTED_HOSTNAME:-user-Super-Server}
UPSTREAM_COMMIT=be2324829b0e91c8fd10a74d4b43714fde6676e1
RUNNER="$TOOL_REPO/scripts/run_in_place_ttt_2x5090_ruler_matrix.sh"
UNDERLYING_AUDITOR="$TOOL_REPO/scripts/audit_in_place_ttt_2x5090_ruler_matrix.py"
SUMMARIZER="$TOOL_REPO/scripts/summarize_in_place_ttt_2x5090_ruler_matrix.py"
WRAPPER_AUDITOR="$TOOL_REPO/scripts/audit_in_place_ttt_qwen3_4b_base_ruler_matrix.py"
CORRECTED_RESOURCE_AUDIT="$RESOURCE_WATCH_ROOT/corrected_independent_audit.json"

if [[ "$(hostname)" != "$EXPECTED_HOSTNAME" ]]; then
  echo "corrected Base RULER matrix is restricted to hku-gpu2" >&2; exit 2
fi
if [[ -e "$OUTPUT_ROOT" || -e "${OUTPUT_ROOT}.partial" || -e "$WATCH_ROOT" ]]; then
  echo "refusing to reuse corrected RULER matrix roots" >&2; exit 3
fi
if [[ "$POLL_SECONDS" -lt 10 || "$MAX_WAIT_SECONDS" -lt "$POLL_SECONDS" ]]; then
  echo "invalid watcher interval/budget" >&2; exit 2
fi
test "$(git -C "$UPSTREAM_REPO" rev-parse HEAD)" = "$UPSTREAM_COMMIT"
test -z "$(git -C "$UPSTREAM_REPO" status --short)"
test "$(git -C "$TOOL_REPO" rev-parse HEAD)" = "$TOOL_COMMIT"
test -z "$(git -C "$TOOL_REPO" status --short)"
for path in "$PYTHON_BIN" "$TOKENIZER/tokenizer.json" "$MODEL_IDENTITY" \
  "$PUBLIC_CORPUS_PLAN" "$EVALUATION_PLAN" "$CORRECTED_PLAN" \
  "$PAPER_INFERENCE_ADDENDUM" "$RUNNER" "$UNDERLYING_AUDITOR" \
  "$SUMMARIZER" "$WRAPPER_AUDITOR"; do test -s "$path"; done

mkdir -p "$WATCH_ROOT"
sha256sum "$0" "$RUNNER" "$UNDERLYING_AUDITOR" "$SUMMARIZER" \
  "$WRAPPER_AUDITOR" "$TOKENIZER/tokenizer.json" "$MODEL_IDENTITY" \
  "$PUBLIC_CORPUS_PLAN" "$EVALUATION_PLAN" "$CORRECTED_PLAN" \
  "$PAPER_INFERENCE_ADDENDUM" > "$WATCH_ROOT/frozen_inputs.sha256"

write_state() {
  printf '%s\n' "$1" > "$WATCH_ROOT/state.txt.partial"
  mv "$WATCH_ROOT/state.txt.partial" "$WATCH_ROOT/state.txt"
  date --iso-8601=seconds > "$WATCH_ROOT/state_time.txt.partial"
  mv "$WATCH_ROOT/state_time.txt.partial" "$WATCH_ROOT/state_time.txt"
}

snapshot() {
  local prefix=$1
  date --iso-8601=seconds > "$WATCH_ROOT/${prefix}_time.txt"
  if [[ -f "$RESOURCE_WATCH_ROOT/state.txt" ]]; then
    cp "$RESOURCE_WATCH_ROOT/state.txt" "$WATCH_ROOT/${prefix}_resource_state.txt"
  else
    : > "$WATCH_ROOT/${prefix}_resource_state.txt"
  fi
  tmux list-panes -t "$RESOURCE_SESSION" \
    -F '#{pane_pid} #{pane_dead} #{pane_current_command}' \
    > "$WATCH_ROOT/${prefix}_resource_handle.txt" 2>&1 || true
}

resource_valid() {
  "$PYTHON_BIN" - "$RESOURCE_ROOT/independent_resource_ladder_audit.json" \
    "$CORRECTED_RESOURCE_AUDIT" "$CORRECTED_TRAINING_AUDIT" <<'PY'
import hashlib, json, pathlib, sys
underlying_path, corrected_path, training_path = map(pathlib.Path, sys.argv[1:])
underlying = json.loads(underlying_path.read_text())
corrected = json.loads(corrected_path.read_text())
expected = {f"{suffix}_{arm}" for suffix in ("4k", "32k", "64k", "128k")
            for arm in ("no_ttt", "ttt")}
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
if (underlying.get("protocol")
        != "in_place_ttt_2x5090_ruler_resource_ladder_audit_v1"
        or underlying.get("passed") is not True or underlying.get("errors") != []
        or set(underlying.get("boundaries", {})) != expected
        or corrected.get("protocol")
        != "in_place_ttt_qwen3_4b_base_ruler_resource_audit_v1"
        or corrected.get("passed") is not True or corrected.get("errors") != []
        or corrected.get("underlying_resource_audit_sha256") != sha(underlying_path)
        or corrected.get("corrected_training_audit_sha256") != sha(training_path)):
    raise SystemExit("corrected RULER resource chain is invalid")
PY
}

started=$(date +%s)
while true; do
  write_state waiting_for_corrected_ruler_resource
  snapshot last
  resource_state=$(cat "$RESOURCE_WATCH_ROOT/state.txt" 2>/dev/null || true)
  case "$resource_state" in
    completed_corrected_ruler_resource)
      test "$(cat "$RESOURCE_ROOT/state.txt")" = completed_ruler_resource_ladder
      test -s "$RESOURCE_ROOT/independent_resource_ladder_audit.json"
      test -s "$CORRECTED_RESOURCE_AUDIT"
      test -s "$CORRECTED_TRAINING_AUDIT"
      resource_valid
      sha256sum "$RESOURCE_ROOT/independent_resource_ladder_audit.json" \
        "$CORRECTED_RESOURCE_AUDIT" "$CORRECTED_TRAINING_AUDIT" \
        > "$WATCH_ROOT/released_resource.sha256"
      break
      ;;
    failed_*|invalid_*|timed_out_*|upstream_*|corrected_formal_stage2_handle_missing)
      write_state "upstream_${resource_state}"; exit 4 ;;
  esac
  if ! tmux has-session -t "$RESOURCE_SESSION" 2>/dev/null; then
    snapshot missing_handle
    write_state corrected_ruler_resource_handle_missing
    exit 5
  fi
  now=$(date +%s)
  [[ $((now - started)) -lt "$MAX_WAIT_SECONDS" ]] || {
    write_state timed_out_waiting_for_corrected_ruler_resource; exit 6;
  }
  sleep "$POLL_SECONDS"
done

write_state launching_corrected_ruler_matrix
set +e
env UPSTREAM_REPO="$UPSTREAM_REPO" TOOL_REPO="$TOOL_REPO" \
  TOOL_COMMIT="$TOOL_COMMIT" PYTHON_BIN="$PYTHON_BIN" TOKENIZER="$TOKENIZER" \
  TRAINING_ROOT="$TRAINING_ROOT" RESOURCE_ROOT="$RESOURCE_ROOT" \
  OUTPUT_ROOT="$OUTPUT_ROOT" PUBLIC_CORPUS_PLAN="$PUBLIC_CORPUS_PLAN" \
  EVALUATION_PLAN="$EVALUATION_PLAN" \
  PAPER_INFERENCE_ADDENDUM="$PAPER_INFERENCE_ADDENDUM" \
  EXPECTED_HOSTNAME="$EXPECTED_HOSTNAME" bash "$RUNNER"
status=$?
set -e
printf '%s\n' "$status" > "$WATCH_ROOT/matrix_runner_exit_code.txt"
[[ "$status" -eq 0 ]] || { write_state failed_corrected_ruler_matrix; exit 7; }

write_state auditing_corrected_ruler_matrix
"$PYTHON_BIN" "$WRAPPER_AUDITOR" \
  --root "$OUTPUT_ROOT" --training-root "$TRAINING_ROOT" \
  --corrected-training-audit "$CORRECTED_TRAINING_AUDIT" \
  --resource-root "$RESOURCE_ROOT" \
  --corrected-resource-audit "$CORRECTED_RESOURCE_AUDIT" \
  --model-identity "$MODEL_IDENTITY" --tokenizer "$TOKENIZER" \
  --public-corpus-plan "$PUBLIC_CORPUS_PLAN" \
  --evaluation-plan "$EVALUATION_PLAN" --corrected-plan "$CORRECTED_PLAN" \
  --paper-inference-addendum "$PAPER_INFERENCE_ADDENDUM" \
  --runner "$RUNNER" --underlying-auditor "$UNDERLYING_AUDITOR" \
  --summarizer "$SUMMARIZER" \
  --output "$WATCH_ROOT/corrected_independent_audit.json" \
  > "$WATCH_ROOT/corrected_independent_audit.stdout"
write_state completed_corrected_ruler_matrix
