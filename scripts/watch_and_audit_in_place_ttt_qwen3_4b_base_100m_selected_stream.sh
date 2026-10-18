#!/usr/bin/env bash
set -euo pipefail

TOOL_REPO=${TOOL_REPO:?set the clean CBF-TTT checkout}
TOOL_COMMIT=${TOOL_COMMIT:?set the exact tool commit}
PYTHON_BIN=${PYTHON_BIN:?set the official training Python}
CORPUS_ROOT=${CORPUS_ROOT:?set the corrected published corpus root}
CORPUS_RUN_ROOT=${CORPUS_RUN_ROOT:?set the corrected corpus wrapper root}
CORPUS_SESSION=${CORPUS_SESSION:?set the corrected corpus tmux handle}
MODEL_IDENTITY=${MODEL_IDENTITY:?set the passing Base identity JSON}
OUTPUT_ROOT=${OUTPUT_ROOT:?set a new selected-stream output root}
POLL_SECONDS=${POLL_SECONDS:-60}
MAX_WAIT_SECONDS=${MAX_WAIT_SECONDS:-604800}
EXPECTED_HOSTNAME=${EXPECTED_HOSTNAME:-user-Super-Server}

PLAN="$TOOL_REPO/IN_PLACE_TTT_BASELINE_QWEN3_4B_BASE_SELECTED_STREAM_ADDENDUM.md"
REPLAY="$TOOL_REPO/scripts/replay_in_place_ttt_qwen3_4b_base_100m_selected_stream.py"
AUDITOR="$TOOL_REPO/scripts/audit_in_place_ttt_qwen3_4b_base_100m_selected_stream.py"
STAGE1_NO="$TOOL_REPO/configs/baseline_reproduction/qwen3_4b_2x5090_100m_stage1_no_ttt.yaml"
STAGE1_TTT="$TOOL_REPO/configs/baseline_reproduction/qwen3_4b_2x5090_100m_stage1_ttt.yaml"
STAGE2_NO="$TOOL_REPO/configs/baseline_reproduction/qwen3_4b_2x5090_100m_stage2_no_ttt.yaml"
STAGE2_TTT="$TOOL_REPO/configs/baseline_reproduction/qwen3_4b_2x5090_100m_stage2_ttt.yaml"
CORPUS_AUDIT="$CORPUS_ROOT/independent_audit.json"
REPLAY_RESULT="$OUTPUT_ROOT/selected_stream_replay.json"
AUDIT_RESULT="$OUTPUT_ROOT/independent_audit.json"

if [[ "$(hostname)" != "$EXPECTED_HOSTNAME" ]]; then
  echo "selected-stream audit is restricted to hku-gpu2" >&2; exit 2
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "refusing to reuse selected-stream root $OUTPUT_ROOT" >&2; exit 3
fi
if [[ "$POLL_SECONDS" -lt 10 || "$MAX_WAIT_SECONDS" -lt "$POLL_SECONDS" ]]; then
  echo "invalid wait interval or budget" >&2; exit 2
fi
test "$(git -C "$TOOL_REPO" rev-parse HEAD)" = "$TOOL_COMMIT"
test -z "$(git -C "$TOOL_REPO" status --short)"
for path in "$PYTHON_BIN" "$MODEL_IDENTITY" "$PLAN" "$REPLAY" "$AUDITOR" \
  "$STAGE1_NO" "$STAGE1_TTT" "$STAGE2_NO" "$STAGE2_TTT"; do test -s "$path"; done

mkdir -p "$OUTPUT_ROOT"
sha256sum "$0" "$PLAN" "$REPLAY" "$AUDITOR" "$MODEL_IDENTITY" \
  "$STAGE1_NO" "$STAGE1_TTT" "$STAGE2_NO" "$STAGE2_TTT" \
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
  write_state failed_selected_stream_audit
  exit "$code"
}
trap on_error ERR

started=$(date +%s)
while true; do
  write_state waiting_for_corrected_base_corpus
  corpus_status=$(cat "$CORPUS_RUN_ROOT/status.txt" 2>/dev/null || true)
  printf '%s\n' "$corpus_status" > "$OUTPUT_ROOT/last_corpus_status.txt"
  if [[ "$corpus_status" == COMPLETE ]]; then
    break
  fi
  if [[ "$corpus_status" == FAILED_* ]]; then
    echo "corrected corpus failed in state $corpus_status" >&2; false
  fi
  if ! tmux has-session -t "$CORPUS_SESSION" 2>/dev/null; then
    sleep 5
    corpus_status=$(cat "$CORPUS_RUN_ROOT/status.txt" 2>/dev/null || true)
    [[ "$corpus_status" == COMPLETE ]] || {
      echo "corpus handle disappeared in state $corpus_status" >&2; false;
    }
    break
  fi
  now=$(date +%s)
  if [[ $((now - started)) -ge "$MAX_WAIT_SECONDS" ]]; then
    echo "timed out waiting for corrected Base corpus" >&2; false
  fi
  sleep "$POLL_SECONDS"
done

write_state verifying_corrected_base_corpus
test -s "$CORPUS_ROOT/manifest.json"
test -s "$CORPUS_AUDIT"
test ! -e "${CORPUS_ROOT}.partial"
env CUDA_VISIBLE_DEVICES= "$PYTHON_BIN" - "$CORPUS_ROOT/manifest.json" "$CORPUS_AUDIT" <<'PY'
import hashlib, json, pathlib, sys
manifest_path, audit_path = map(pathlib.Path, sys.argv[1:])
manifest, audit = json.loads(manifest_path.read_text()), json.loads(audit_path.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert manifest["profile"] == "full"
assert manifest["totals"]["records"] == 724800
assert manifest["totals"]["training_tokens"] == 34999369728
assert audit["passed"] is True and all(audit["checks"].values())
assert audit["bindings"]["manifest_sha256"] == sha(manifest_path)
PY
sha256sum "$CORPUS_ROOT/manifest.json" "$CORPUS_AUDIT" \
  "$CORPUS_ROOT/stage1_32k.provenance.jsonl" \
  "$CORPUS_ROOT/stage2_128k.provenance.jsonl" \
  > "$OUTPUT_ROOT/released_corpus_inputs.sha256"

common_args=(
  --corpus-root "$CORPUS_ROOT" --corpus-audit "$CORPUS_AUDIT"
  --model-identity "$MODEL_IDENTITY" --plan "$PLAN"
  --stage1-no-ttt-config "$STAGE1_NO" --stage1-ttt-config "$STAGE1_TTT"
  --stage2-no-ttt-config "$STAGE2_NO" --stage2-ttt-config "$STAGE2_TTT"
  --expected-hostname "$EXPECTED_HOSTNAME"
)

write_state replaying_native_selected_stream
env CUDA_VISIBLE_DEVICES= "$PYTHON_BIN" "$REPLAY" "${common_args[@]}" \
  --output "$REPLAY_RESULT" > "$OUTPUT_ROOT/replay.stdout"

write_state independently_auditing_selected_stream
env CUDA_VISIBLE_DEVICES= "$PYTHON_BIN" "$AUDITOR" \
  --replay "$REPLAY_RESULT" --replay-script "$REPLAY" "${common_args[@]}" \
  --output "$AUDIT_RESULT" > "$OUTPUT_ROOT/audit.stdout"
env CUDA_VISIBLE_DEVICES= "$PYTHON_BIN" - "$REPLAY_RESULT" "$AUDIT_RESULT" <<'PY'
import json, pathlib, sys
replay, audit = [json.loads(pathlib.Path(path).read_text()) for path in sys.argv[1:]]
assert replay["passed"] is True
assert audit["passed"] is True and all(audit["checks"].values())
PY

sha256sum "$REPLAY_RESULT" "$AUDIT_RESULT" > "$OUTPUT_ROOT/final_outputs.sha256"
write_state completed_selected_stream_audit
trap - ERR
