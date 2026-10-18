#!/usr/bin/env bash
set -Eeuo pipefail

UPSTREAM_REPO=${UPSTREAM_REPO:?set the clean official checkout}
TOOL_REPO=${TOOL_REPO:?set the clean CBF-TTT checkout}
TOOL_COMMIT=${TOOL_COMMIT:?set the exact tool commit}
PYTHON_BIN=${PYTHON_BIN:?set the isolated OpenCompass Python}
TOKENIZER=${TOKENIZER:?set the original Qwen3-4B tokenizer directory}
TRAINING_ROOT=${TRAINING_ROOT:?set the completed paired formal Stage-2 root}
MATRIX_ROOT=${MATRIX_ROOT:?set the completed audited 4K-128K matrix root}
RESOURCE_ROOT=${RESOURCE_ROOT:?set the completed audited 256K resource root}
RELEASE=${RELEASE:?set the explicit 256K formal release JSON}
OUTPUT_ROOT=${OUTPUT_ROOT:?set a new paired formal 256K root}
PUBLIC_CORPUS_PLAN=${PUBLIC_CORPUS_PLAN:-$TOOL_REPO/IN_PLACE_TTT_BASELINE_PUBLIC_CORPUS_PLAN.md}
EVALUATION_PLAN=${EVALUATION_PLAN:-$TOOL_REPO/IN_PLACE_TTT_BASELINE_2X5090_RULER_PLAN.md}
EXTRAPOLATION_PLAN=${EXTRAPOLATION_PLAN:-$TOOL_REPO/IN_PLACE_TTT_BASELINE_2X5090_RULER_256K_PLAN.md}
FORMAL_PLAN=${FORMAL_PLAN:-$TOOL_REPO/IN_PLACE_TTT_BASELINE_2X5090_RULER_256K_FORMAL_PLAN.md}
EXPECTED_HOSTNAME=${EXPECTED_HOSTNAME:-user-Super-Server}
UPSTREAM_COMMIT=be2324829b0e91c8fd10a74d4b43714fde6676e1

if [[ "$(hostname)" != "$EXPECTED_HOSTNAME" ]]; then
  echo "formal RULER-256K is restricted to hku-gpu2" >&2; exit 2
fi
if [[ -e "$OUTPUT_ROOT" || -e "${OUTPUT_ROOT}.partial" ]]; then
  echo "refusing to overwrite formal 256K output" >&2; exit 3
fi
test "$(git -C "$TOOL_REPO" rev-parse HEAD)" = "$TOOL_COMMIT"
test -z "$(git -C "$TOOL_REPO" status --short)"
test "$(git -C "$UPSTREAM_REPO" rev-parse HEAD)" = "$UPSTREAM_COMMIT"
test -z "$(git -C "$UPSTREAM_REPO" status --short)"
test -x "$PYTHON_BIN"; test -s "$TOKENIZER/tokenizer.json"
test "$(cat "$TRAINING_ROOT/state.txt")" = completed_100m_stage2
test "$(cat "$MATRIX_ROOT/state.txt")" = completed_ruler_matrix
test "$(cat "$RESOURCE_ROOT/state.txt")" = completed_ruler_256k_resource
for path in "$TRAINING_ROOT/independent_audit.json" \
  "$MATRIX_ROOT/independent_matrix_audit.json" \
  "$RESOURCE_ROOT/independent_resource_audit.json" "$RELEASE" \
  "$PUBLIC_CORPUS_PLAN" "$EVALUATION_PLAN" "$EXTRAPOLATION_PLAN" \
  "$FORMAL_PLAN"; do test -s "$path"; done

readarray -t models < <("$PYTHON_BIN" - "$TRAINING_ROOT" "$MATRIX_ROOT" \
  "$RESOURCE_ROOT" "$RELEASE" <<'PY'
import hashlib, json, pathlib, sys
training_root, matrix_root, resource_root, release_path = map(pathlib.Path, sys.argv[1:])
training = json.loads((training_root / "independent_audit.json").read_text())
matrix = json.loads((matrix_root / "independent_matrix_audit.json").read_text())
resource_path = resource_root / "independent_resource_audit.json"
resource = json.loads(resource_path.read_text())
release = json.loads(release_path.read_text())
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
if (training.get("protocol") != "in_place_ttt_2x5090_100m_stage2_audit_v1"
        or training.get("passed") is not True or training.get("errors") != []
        or matrix.get("protocol") != "in_place_ttt_2x5090_ruler_matrix_audit_v1"
        or matrix.get("passed") is not True or matrix.get("errors") != []
        or resource.get("protocol")
        != "in_place_ttt_2x5090_ruler_256k_resource_audit_v1"
        or resource.get("passed") is not True or resource.get("errors") != []
        or resource.get("formal_256k_released") is not True
        or release.get("protocol")
        != "in_place_ttt_2x5090_ruler_256k_formal_release_v1"
        or release.get("accepted") is not True
        or release.get("resource_audit_sha256") != sha(resource_path)):
    raise SystemExit("formal 256K prerequisites did not pass")
for arm in ("no_ttt", "ttt"):
    record = training.get("arms", {}).get(arm, {})
    model = training_root / arm / record.get("final_hf_relative_path", "")
    if not (model / "config.json").is_file():
        raise SystemExit(f"missing final HF checkpoint for {arm}")
    print(model)
PY
)
test ${#models[@]} -eq 2

staging=${OUTPUT_ROOT}.partial
mkdir -p "$staging"
finish() {
  rc=$?; trap - EXIT
  if [[ $rc -eq 0 ]]; then
    printf 'COMPLETED\nfinished_utc=%s\nexit_code=0\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$staging/status.txt"
    mv "$staging" "$OUTPUT_ROOT"
  else
    printf 'FAILED\nfinished_utc=%s\nexit_code=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" > "$staging/status.txt"
  fi
  exit "$rc"
}
trap finish EXIT
printf 'RUNNING\nstarted_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "$staging/status.txt"

"$PYTHON_BIN" - "$TOOL_REPO" "$TRAINING_ROOT" "$MATRIX_ROOT" \
  "$RESOURCE_ROOT" "$RELEASE" "$PUBLIC_CORPUS_PLAN" "$EVALUATION_PLAN" \
  "$EXTRAPOLATION_PLAN" "$FORMAL_PLAN" "$staging/formal_metadata.json" <<'PY'
import hashlib, json, pathlib, socket, subprocess, sys
(tool, training, matrix, resource, release, public, evaluation, extrapolation,
 formal, output) = map(pathlib.Path, sys.argv[1:])
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
metadata = {
    "protocol": "in_place_ttt_2x5090_ruler_256k_formal_run_v1",
    "hostname": socket.gethostname(),
    "tool_commit": subprocess.check_output(
        ["git", "-C", tool, "rev-parse", "HEAD"], text=True).strip(),
    "tool_clean": not subprocess.check_output(
        ["git", "-C", tool, "status", "--short"], text=True).strip(),
    "training_root": str(training.resolve()),
    "matrix_root": str(matrix.resolve()),
    "resource_root": str(resource.resolve()),
    "context_length": 262144,
    "training_audit_sha256": sha(training / "independent_audit.json"),
    "matrix_audit_sha256": sha(matrix / "independent_matrix_audit.json"),
    "resource_audit_sha256": sha(resource / "independent_resource_audit.json"),
    "release_sha256": sha(release),
    "public_corpus_plan_sha256": sha(public),
    "evaluation_plan_sha256": sha(evaluation),
    "extrapolation_plan_sha256": sha(extrapolation),
    "formal_plan_sha256": sha(formal),
    "runner_sha256": sha(tool / "scripts/run_in_place_ttt_2x5090_ruler_256k.sh"),
    "single_runner_sha256": sha(tool / "scripts/run_in_place_ttt_2x5090_ruler_single.sh"),
    "summarizer_sha256": sha(tool / "scripts/summarize_in_place_ttt_2x5090_ruler_256k.py"),
    "auditor_sha256": sha(tool / "scripts/audit_in_place_ttt_2x5090_ruler_256k.py"),
}
output.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
PY

pids=()
for arm in no_ttt ttt; do
  if [[ "$arm" == no_ttt ]]; then model=${models[0]}; gpu=0
  else model=${models[1]}; gpu=1; fi
  (
    export UPSTREAM_REPO TOOL_REPO TOOL_COMMIT PYTHON_BIN TOKENIZER
    export PUBLIC_CORPUS_PLAN EVALUATION_PLAN TRAINING_ROOT
    export MODEL="$model" RULER_RUN_KIND=formal RULER_NUM_SAMPLES=100
    export OUTPUT_ROOT="$staging/$arm"
    bash "$TOOL_REPO/scripts/run_in_place_ttt_2x5090_ruler_single.sh" \
      262144 "$arm" "$gpu"
  ) > "$staging/${arm}.launcher.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
if [[ $failed -ne 0 ]]; then echo "paired formal RULER-256K failed" >&2; exit 4; fi

"$PYTHON_BIN" "$TOOL_REPO/scripts/summarize_in_place_ttt_2x5090_ruler_256k.py" \
  --root "$staging" --output "$staging/summary.json" > "$staging/summary.log"
printf 'auditing_ruler_256k\n' > "$staging/state.txt"
"$PYTHON_BIN" "$TOOL_REPO/scripts/audit_in_place_ttt_2x5090_ruler_256k.py" \
  --root "$staging" --training-root "$TRAINING_ROOT" \
  --matrix-root "$MATRIX_ROOT" --resource-root "$RESOURCE_ROOT" \
  --release "$RELEASE" --public-corpus-plan "$PUBLIC_CORPUS_PLAN" \
  --evaluation-plan "$EVALUATION_PLAN" \
  --extrapolation-plan "$EXTRAPOLATION_PLAN" --formal-plan "$FORMAL_PLAN" \
  --runner "$TOOL_REPO/scripts/run_in_place_ttt_2x5090_ruler_256k.sh" \
  --output "$staging/independent_audit.json" > "$staging/independent_audit.log"
printf 'completed_ruler_256k_formal\n' > "$staging/state.txt"
