#!/usr/bin/env bash
set -euo pipefail

TOOL_REPO=${TOOL_REPO:?set the clean CBF-TTT checkout}
TOOL_COMMIT=${TOOL_COMMIT:?set the exact tool commit}
PYTHON_BIN=${PYTHON_BIN:?set the isolated OpenCompass Python}
RESOURCE_ROOT=${RESOURCE_ROOT:?set the corrected 256K resource root}
RESOURCE_WATCH_ROOT=${RESOURCE_WATCH_ROOT:?set its corrected wrapper root}
RESOURCE_SESSION=${RESOURCE_SESSION:?set its live tmux session}
MODEL_IDENTITY=${MODEL_IDENTITY:?set the passing Base identity JSON}
TARGET_RELEASE=${TARGET_RELEASE:?set the future formal release JSON}
OUTPUT_ROOT=${OUTPUT_ROOT:?set a new non-authorizing review root}
POLL_SECONDS=${POLL_SECONDS:-60}
MAX_WAIT_SECONDS=${MAX_WAIT_SECONDS:-1209600}
EXPECTED_HOSTNAME=${EXPECTED_HOSTNAME:-user-Super-Server}

PLAN="$TOOL_REPO/IN_PLACE_TTT_BASELINE_QWEN3_4B_BASE_RULER_256K_RELEASE_ADDENDUM.md"
BUILDER="$TOOL_REPO/scripts/create_in_place_ttt_qwen3_4b_base_ruler_256k_release_review.py"
TRANSITION="$TOOL_REPO/scripts/create_in_place_ttt_qwen3_4b_base_ruler_256k_release_from_review.py"
RELEASE_TOOL="$TOOL_REPO/scripts/create_in_place_ttt_2x5090_ruler_256k_release.py"
RESOURCE_AUDIT="$RESOURCE_ROOT/independent_resource_audit.json"
CORRECTED_AUDIT="$RESOURCE_WATCH_ROOT/corrected_independent_audit.json"
REVIEW="$OUTPUT_ROOT/review.json"

if [[ "$(hostname)" != "$EXPECTED_HOSTNAME" ]]; then
  echo "corrected 256K release review is restricted to hku-gpu2" >&2; exit 2
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "refusing to reuse corrected 256K review root" >&2; exit 3
fi
if [[ -e "$TARGET_RELEASE" || -e "${TARGET_RELEASE}.partial" ]]; then
  echo "target release exists before corrected review" >&2; exit 3
fi
if [[ "$POLL_SECONDS" -lt 10 || "$MAX_WAIT_SECONDS" -lt "$POLL_SECONDS" ]]; then
  echo "invalid wait interval or budget" >&2; exit 2
fi
test "$(git -C "$TOOL_REPO" rev-parse HEAD)" = "$TOOL_COMMIT"
test -z "$(git -C "$TOOL_REPO" status --short)"
for path in "$PYTHON_BIN" "$MODEL_IDENTITY" "$PLAN" "$BUILDER" \
  "$TRANSITION" "$RELEASE_TOOL"; do test -s "$path"; done

mkdir -p "$OUTPUT_ROOT"
sha256sum "$0" "$PLAN" "$BUILDER" "$TRANSITION" "$RELEASE_TOOL" \
  "$MODEL_IDENTITY" > "$OUTPUT_ROOT/frozen_inputs.sha256"
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
  printf 'exit=%s state=%s\n' "$code" \
    "$(cat "$OUTPUT_ROOT/state.txt" 2>/dev/null || true)" >> "$OUTPUT_ROOT/error.log"
  write_state failed_ruler_256k_release_review
  exit "$code"
}
trap on_error ERR

started=$(date +%s)
while true; do
  write_state waiting_for_corrected_ruler_256k_resource
  resource_state=$(cat "$RESOURCE_WATCH_ROOT/state.txt" 2>/dev/null || true)
  printf '%s\n' "$resource_state" > "$OUTPUT_ROOT/last_resource_state.txt"
  tmux list-panes -t "$RESOURCE_SESSION" \
    -F '#{pane_pid} #{pane_dead} #{pane_current_command}' \
    > "$OUTPUT_ROOT/last_resource_handle.txt" 2>&1 || true
  if [[ -e "$TARGET_RELEASE" || -e "${TARGET_RELEASE}.partial" ]]; then
    echo "target release appeared before corrected review" >&2; false
  fi
  case "$resource_state" in
    completed_corrected_ruler_256k_resource) break ;;
    failed_*|invalid_*|timed_out_*|upstream_*|*_handle_missing)
      echo "corrected 256K resource failed: $resource_state" >&2; false ;;
  esac
  tmux has-session -t "$RESOURCE_SESSION" 2>/dev/null || {
    echo "corrected 256K resource handle missing" >&2; false;
  }
  now=$(date +%s)
  [[ $((now - started)) -lt "$MAX_WAIT_SECONDS" ]] || {
    echo "timed out waiting for corrected 256K resource" >&2; false;
  }
  sleep "$POLL_SECONDS"
done

write_state building_ruler_256k_release_review
test "$(cat "$RESOURCE_ROOT/state.txt")" = completed_ruler_256k_resource
test -s "$RESOURCE_AUDIT"; test -s "$CORRECTED_AUDIT"
env CUDA_VISIBLE_DEVICES= "$PYTHON_BIN" "$BUILDER" \
  --resource-audit "$RESOURCE_AUDIT" \
  --corrected-resource-audit "$CORRECTED_AUDIT" \
  --model-identity "$MODEL_IDENTITY" --release-tool "$RELEASE_TOOL" \
  --review-builder "$BUILDER" --transition-tool "$TRANSITION" --plan "$PLAN" \
  --target-release "$TARGET_RELEASE" --output "$REVIEW" \
  --expected-hostname "$EXPECTED_HOSTNAME" > "$OUTPUT_ROOT/review.stdout"

env CUDA_VISIBLE_DEVICES= "$PYTHON_BIN" - "$REVIEW" "$RESOURCE_AUDIT" \
  "$CORRECTED_AUDIT" "$MODEL_IDENTITY" "$TARGET_RELEASE" <<'PY'
import hashlib, json, pathlib, sys
review_path, resource, corrected, identity, target = map(pathlib.Path, sys.argv[1:])
payload = json.loads(review_path.read_text())
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
assert payload["protocol"] == "in_place_ttt_qwen3_4b_base_ruler_256k_release_review_v1"
assert payload["ready_for_human_review"] is True
assert payload["formal_release_created"] is False
assert payload["target_release_path"] == str(target.resolve())
assert not target.exists() and not target.with_name(target.name + ".partial").exists()
assert payload["bindings"]["resource_audit_sha256"] == sha(resource)
assert payload["bindings"]["corrected_resource_audit_sha256"] == sha(corrected)
assert payload["bindings"]["model_identity_sha256"] == sha(identity)
assert set(payload["required_human_fields"]) == {"accept", "reviewed_by", "decision_note"}
PY
sha256sum "$RESOURCE_AUDIT" "$CORRECTED_AUDIT" "$MODEL_IDENTITY" "$REVIEW" \
  > "$OUTPUT_ROOT/final_outputs.sha256"
write_state completed_ruler_256k_release_review
trap - ERR
