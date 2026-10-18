#!/usr/bin/env bash
set -euo pipefail

UPSTREAM_REPO=${UPSTREAM_REPO:?set the clean official checkout}
TOOL_REPO=${TOOL_REPO:?set the clean CBF-TTT checkout}
TOOL_COMMIT=${TOOL_COMMIT:?set the exact tool commit}
PYTHON_BIN=${PYTHON_BIN:?set the isolated OpenCompass Python}
TOKENIZER=${TOKENIZER:?set the exact Qwen3-4B-Base tokenizer}
MODEL_IDENTITY=${MODEL_IDENTITY:?set the passing Base identity JSON}
TRAINING_ROOT=${TRAINING_ROOT:?set the formal Stage-2 root}
CORRECTED_TRAINING_AUDIT=${CORRECTED_TRAINING_AUDIT:?set its corrected audit}
MATRIX_ROOT=${MATRIX_ROOT:?set the primary matrix root}
CORRECTED_MATRIX_AUDIT=${CORRECTED_MATRIX_AUDIT:?set its corrected audit}
RESOURCE_ROOT=${RESOURCE_ROOT:?set the 256K resource root}
CORRECTED_RESOURCE_AUDIT=${CORRECTED_RESOURCE_AUDIT:?set its corrected audit}
REVIEW_ROOT=${REVIEW_ROOT:?set the non-authorizing review root}
REVIEW_SESSION=${REVIEW_SESSION:?set its tmux session}
RELEASE=${RELEASE:?set the future human-accepted release}
OUTPUT_ROOT=${OUTPUT_ROOT:?set a new formal 256K output root}
WATCH_ROOT=${WATCH_ROOT:-${OUTPUT_ROOT}.watch}
PUBLIC_CORPUS_PLAN=${PUBLIC_CORPUS_PLAN:?set the public-corpus plan}
EVALUATION_PLAN=${EVALUATION_PLAN:?set the RULER evaluation plan}
EXTRAPOLATION_PLAN=${EXTRAPOLATION_PLAN:?set the generic 256K plan}
FORMAL_PLAN=${FORMAL_PLAN:?set the generic formal 256K plan}
CORRECTED_256K_PLAN=${CORRECTED_256K_PLAN:?set the corrected Base 256K plan}
RELEASE_ADDENDUM=${RELEASE_ADDENDUM:?set the corrected release addendum}
FORMAL_ADDENDUM=${FORMAL_ADDENDUM:?set the corrected formal addendum}
PAPER_INFERENCE_ADDENDUM=${PAPER_INFERENCE_ADDENDUM:?set the paper clip addendum}
POLL_SECONDS=${POLL_SECONDS:-60}
MAX_WAIT_SECONDS=${MAX_WAIT_SECONDS:-1209600}
EXPECTED_HOSTNAME=${EXPECTED_HOSTNAME:-user-Super-Server}
UPSTREAM_COMMIT=be2324829b0e91c8fd10a74d4b43714fde6676e1

RUNNER="$TOOL_REPO/scripts/run_in_place_ttt_2x5090_ruler_256k.sh"
AUDITOR="$TOOL_REPO/scripts/audit_in_place_ttt_qwen3_4b_base_ruler_256k.py"
UNDERLYING_AUDITOR="$TOOL_REPO/scripts/audit_in_place_ttt_2x5090_ruler_256k.py"
SINGLE_RUNNER="$TOOL_REPO/scripts/run_in_place_ttt_2x5090_ruler_single.sh"
SUMMARIZER="$TOOL_REPO/scripts/summarize_in_place_ttt_2x5090_ruler_256k.py"
REVIEW_BUILDER="$TOOL_REPO/scripts/create_in_place_ttt_qwen3_4b_base_ruler_256k_release_review.py"
TRANSITION="$TOOL_REPO/scripts/create_in_place_ttt_qwen3_4b_base_ruler_256k_release_from_review.py"
RELEASE_TOOL="$TOOL_REPO/scripts/create_in_place_ttt_2x5090_ruler_256k_release.py"
REVIEW_PACKET="$REVIEW_ROOT/review.json"

if [[ "$(hostname)" != "$EXPECTED_HOSTNAME" ]]; then
  echo "corrected formal RULER-256K is restricted to hku-gpu2" >&2; exit 2
fi
if [[ -e "$OUTPUT_ROOT" || -e "${OUTPUT_ROOT}.partial" || -e "$WATCH_ROOT" ]]; then
  echo "refusing to reuse corrected formal 256K roots" >&2; exit 3
fi
if [[ "$POLL_SECONDS" -lt 10 || "$MAX_WAIT_SECONDS" -lt "$POLL_SECONDS" ]]; then
  echo "invalid wait interval or budget" >&2; exit 2
fi
test "$(git -C "$UPSTREAM_REPO" rev-parse HEAD)" = "$UPSTREAM_COMMIT"
test -z "$(git -C "$UPSTREAM_REPO" status --short)"
test "$(git -C "$TOOL_REPO" rev-parse HEAD)" = "$TOOL_COMMIT"
test -z "$(git -C "$TOOL_REPO" status --short)"
for path in "$PYTHON_BIN" "$TOKENIZER/tokenizer.json" "$MODEL_IDENTITY" \
  "$PUBLIC_CORPUS_PLAN" "$EVALUATION_PLAN" "$EXTRAPOLATION_PLAN" \
  "$FORMAL_PLAN" "$CORRECTED_256K_PLAN" \
  "$RELEASE_ADDENDUM" "$FORMAL_ADDENDUM" "$PAPER_INFERENCE_ADDENDUM" \
  "$RUNNER" "$AUDITOR" "$UNDERLYING_AUDITOR" "$SINGLE_RUNNER" \
  "$SUMMARIZER" "$REVIEW_BUILDER" "$TRANSITION" "$RELEASE_TOOL"; do
  test -s "$path"
done

mkdir -p "$WATCH_ROOT"
sha256sum "$0" "$RUNNER" "$AUDITOR" "$UNDERLYING_AUDITOR" \
  "$SINGLE_RUNNER" "$SUMMARIZER" "$REVIEW_BUILDER" "$TRANSITION" \
  "$RELEASE_TOOL" "$MODEL_IDENTITY" "$TOKENIZER/tokenizer.json" \
  "$PUBLIC_CORPUS_PLAN" "$EVALUATION_PLAN" "$EXTRAPOLATION_PLAN" \
  "$FORMAL_PLAN" "$CORRECTED_256K_PLAN" "$RELEASE_ADDENDUM" \
  "$FORMAL_ADDENDUM" "$PAPER_INFERENCE_ADDENDUM" \
  > "$WATCH_ROOT/frozen_inputs.sha256"

write_state() {
  printf '%s\n' "$1" > "$WATCH_ROOT/state.txt.partial"
  mv "$WATCH_ROOT/state.txt.partial" "$WATCH_ROOT/state.txt"
  date --iso-8601=seconds > "$WATCH_ROOT/state_time.txt.partial"
  mv "$WATCH_ROOT/state_time.txt.partial" "$WATCH_ROOT/state_time.txt"
}
validate_release() {
  env CUDA_VISIBLE_DEVICES= "$PYTHON_BIN" - "$TOOL_REPO" "$REVIEW_PACKET" \
    "$RELEASE" "$RELEASE_ADDENDUM" "$RESOURCE_ROOT/independent_resource_audit.json" \
    "$CORRECTED_RESOURCE_AUDIT" "$MODEL_IDENTITY" <<'PY'
import hashlib, json, pathlib, sys
tool, packet_path, release_path, plan, resource, corrected, identity = map(pathlib.Path, sys.argv[1:])
sys.path.insert(0, str(tool))
from scripts import create_in_place_ttt_qwen3_4b_base_ruler_256k_release_review as review_module
packet = json.loads(packet_path.read_text())
release = json.loads(release_path.read_text())
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
recomputed = review_module.validated_review(
    resource_audit=resource, corrected_resource_audit=corrected,
    model_identity=identity,
    release_tool=tool / "scripts/create_in_place_ttt_2x5090_ruler_256k_release.py")
if (packet.get("protocol") != review_module.PROTOCOL
        or packet.get("ready_for_human_review") is not True
        or packet.get("formal_release_created") is not False
        or packet.get("target_release_path") != str(release_path.resolve())
        or packet.get("review") != recomputed
        or release.get("protocol") != "in_place_ttt_2x5090_ruler_256k_formal_release_v1"
        or release.get("corrected_base_release_protocol")
        != "in_place_ttt_qwen3_4b_base_ruler_256k_formal_release_v1"
        or release.get("accepted") is not True
        or release.get("model_revision") != review_module.MODEL_REVISION
        or release.get("resource_audit_sha256") != sha(resource)
        or release.get("corrected_resource_audit_sha256") != sha(corrected)
        or release.get("review_packet_sha256") != sha(packet_path)
        or release.get("release_transition_plan_sha256") != sha(plan)
        or not str(release.get("reviewed_by", "")).strip()
        or not str(release.get("decision_note", "")).strip()):
    raise SystemExit("corrected 256K release or review packet is invalid")
PY
}

started=$(date +%s)
while true; do
  write_state waiting_for_accepted_corrected_ruler_256k_release
  review_state=$(cat "$REVIEW_ROOT/state.txt" 2>/dev/null || true)
  printf '%s\n' "$review_state" > "$WATCH_ROOT/last_review_state.txt"
  tmux list-panes -t "$REVIEW_SESSION" \
    -F '#{pane_pid} #{pane_dead} #{pane_current_command}' \
    > "$WATCH_ROOT/last_review_handle.txt" 2>&1 || true
  case "$review_state" in
    completed_ruler_256k_release_review)
      test -s "$REVIEW_PACKET"
      if [[ -s "$RELEASE" ]]; then validate_release; break; fi
      [[ ! -e "${RELEASE}.partial" ]] || {
        echo "partial corrected 256K release is not acceptable" >&2; exit 4;
      }
      ;;
    failed_*|invalid_*|timed_out_*|upstream_*|*_handle_missing)
      write_state "upstream_${review_state}"; exit 5 ;;
    *)
      tmux has-session -t "$REVIEW_SESSION" 2>/dev/null || {
        write_state corrected_ruler_256k_review_handle_missing; exit 6;
      }
      ;;
  esac
  now=$(date +%s)
  [[ $((now - started)) -lt "$MAX_WAIT_SECONDS" ]] || {
    write_state timed_out_waiting_for_corrected_ruler_256k_release; exit 7;
  }
  sleep "$POLL_SECONDS"
done

for path in "$TRAINING_ROOT/independent_audit.json" \
  "$MATRIX_ROOT/independent_matrix_audit.json" \
  "$RESOURCE_ROOT/independent_resource_audit.json" \
  "$CORRECTED_TRAINING_AUDIT" "$CORRECTED_MATRIX_AUDIT" \
  "$CORRECTED_RESOURCE_AUDIT"; do test -s "$path"; done
sha256sum "$REVIEW_PACKET" "$RELEASE" \
  "$TRAINING_ROOT/independent_audit.json" \
  "$MATRIX_ROOT/independent_matrix_audit.json" \
  "$RESOURCE_ROOT/independent_resource_audit.json" \
  "$CORRECTED_TRAINING_AUDIT" \
  "$CORRECTED_MATRIX_AUDIT" "$CORRECTED_RESOURCE_AUDIT" \
  > "$WATCH_ROOT/accepted_release.sha256"
write_state launching_corrected_formal_ruler_256k
set +e
env UPSTREAM_REPO="$UPSTREAM_REPO" TOOL_REPO="$TOOL_REPO" \
  TOOL_COMMIT="$TOOL_COMMIT" PYTHON_BIN="$PYTHON_BIN" TOKENIZER="$TOKENIZER" \
  TRAINING_ROOT="$TRAINING_ROOT" MATRIX_ROOT="$MATRIX_ROOT" \
  RESOURCE_ROOT="$RESOURCE_ROOT" RELEASE="$RELEASE" OUTPUT_ROOT="$OUTPUT_ROOT" \
  PUBLIC_CORPUS_PLAN="$PUBLIC_CORPUS_PLAN" EVALUATION_PLAN="$EVALUATION_PLAN" \
  EXTRAPOLATION_PLAN="$EXTRAPOLATION_PLAN" FORMAL_PLAN="$FORMAL_PLAN" \
  PAPER_INFERENCE_ADDENDUM="$PAPER_INFERENCE_ADDENDUM" \
  EXPECTED_HOSTNAME="$EXPECTED_HOSTNAME" bash "$RUNNER"
status=$?
set -e
printf '%s\n' "$status" > "$WATCH_ROOT/formal_runner_exit_code.txt"
[[ "$status" -eq 0 ]] || { write_state failed_corrected_formal_ruler_256k; exit 8; }

write_state auditing_corrected_formal_ruler_256k
"$PYTHON_BIN" "$AUDITOR" --root "$OUTPUT_ROOT" \
  --training-root "$TRAINING_ROOT" \
  --corrected-training-audit "$CORRECTED_TRAINING_AUDIT" \
  --matrix-root "$MATRIX_ROOT" --corrected-matrix-audit "$CORRECTED_MATRIX_AUDIT" \
  --resource-root "$RESOURCE_ROOT" \
  --corrected-resource-audit "$CORRECTED_RESOURCE_AUDIT" \
  --review-packet "$REVIEW_PACKET" --release "$RELEASE" \
  --model-identity "$MODEL_IDENTITY" --tokenizer "$TOKENIZER" \
  --public-corpus-plan "$PUBLIC_CORPUS_PLAN" \
  --evaluation-plan "$EVALUATION_PLAN" \
  --extrapolation-plan "$EXTRAPOLATION_PLAN" --formal-plan "$FORMAL_PLAN" \
  --corrected-256k-plan "$CORRECTED_256K_PLAN" \
  --release-addendum "$RELEASE_ADDENDUM" --formal-addendum "$FORMAL_ADDENDUM" \
  --paper-inference-addendum "$PAPER_INFERENCE_ADDENDUM" --runner "$RUNNER" \
  --underlying-auditor "$UNDERLYING_AUDITOR" --single-runner "$SINGLE_RUNNER" \
  --summarizer "$SUMMARIZER" --review-builder "$REVIEW_BUILDER" \
  --transition-tool "$TRANSITION" --release-tool "$RELEASE_TOOL" \
  --output "$WATCH_ROOT/corrected_independent_audit.json" \
  > "$WATCH_ROOT/corrected_independent_audit.stdout"
write_state completed_corrected_formal_ruler_256k
