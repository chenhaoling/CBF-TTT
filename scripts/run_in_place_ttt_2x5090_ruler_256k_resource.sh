#!/usr/bin/env bash
set -Eeuo pipefail

UPSTREAM_REPO=${UPSTREAM_REPO:?set the clean official checkout}
TOOL_REPO=${TOOL_REPO:?set the clean CBF-TTT checkout}
TOOL_COMMIT=${TOOL_COMMIT:?set the exact tool commit}
PYTHON_BIN=${PYTHON_BIN:?set the isolated OpenCompass Python}
TOKENIZER=${TOKENIZER:?set the original Qwen3-4B tokenizer directory}
TRAINING_ROOT=${TRAINING_ROOT:?set the completed paired formal Stage-2 root}
MATRIX_ROOT=${MATRIX_ROOT:?set the completed audited 4K-128K matrix root}
OUTPUT_ROOT=${OUTPUT_ROOT:?set a new paired 256K resource root}
PUBLIC_CORPUS_PLAN=${PUBLIC_CORPUS_PLAN:-$TOOL_REPO/IN_PLACE_TTT_BASELINE_PUBLIC_CORPUS_PLAN.md}
EVALUATION_PLAN=${EVALUATION_PLAN:-$TOOL_REPO/IN_PLACE_TTT_BASELINE_2X5090_RULER_PLAN.md}
EXTRAPOLATION_PLAN=${EXTRAPOLATION_PLAN:-$TOOL_REPO/IN_PLACE_TTT_BASELINE_2X5090_RULER_256K_PLAN.md}
EXPECTED_HOSTNAME=${EXPECTED_HOSTNAME:-user-Super-Server}
UPSTREAM_COMMIT=be2324829b0e91c8fd10a74d4b43714fde6676e1

if [[ "$(hostname)" != "$EXPECTED_HOSTNAME" ]]; then
  echo "2x5090 RULER-256K resource gate is restricted to hku-gpu2" >&2; exit 2
fi
if [[ -e "$OUTPUT_ROOT" || -e "${OUTPUT_ROOT}.partial" ]]; then
  echo "refusing to overwrite 256K resource output" >&2; exit 3
fi
test "$(git -C "$TOOL_REPO" rev-parse HEAD)" = "$TOOL_COMMIT"
test -z "$(git -C "$TOOL_REPO" status --short)"
test "$(git -C "$UPSTREAM_REPO" rev-parse HEAD)" = "$UPSTREAM_COMMIT"
test -z "$(git -C "$UPSTREAM_REPO" status --short)"
test -x "$PYTHON_BIN"; test -s "$TOKENIZER/tokenizer.json"
test "$(cat "$TRAINING_ROOT/state.txt")" = completed_100m_stage2
test "$(cat "$MATRIX_ROOT/state.txt")" = completed_ruler_matrix
for path in "$TRAINING_ROOT/independent_audit.json" \
  "$MATRIX_ROOT/independent_matrix_audit.json" "$PUBLIC_CORPUS_PLAN" \
  "$EVALUATION_PLAN" "$EXTRAPOLATION_PLAN"; do test -s "$path"; done

readarray -t models < <("$PYTHON_BIN" - "$TRAINING_ROOT" "$MATRIX_ROOT" <<'PY'
import json, pathlib, sys
training_root, matrix_root = map(pathlib.Path, sys.argv[1:])
training = json.loads((training_root / "independent_audit.json").read_text())
matrix = json.loads((matrix_root / "independent_matrix_audit.json").read_text())
if (training.get("protocol") != "in_place_ttt_2x5090_100m_stage2_audit_v1"
        or training.get("passed") is not True or training.get("errors") != []
        or matrix.get("protocol") != "in_place_ttt_2x5090_ruler_matrix_audit_v1"
        or matrix.get("passed") is not True or matrix.get("errors") != []):
    raise SystemExit("formal Stage-2 or 4K-128K matrix audit did not pass")
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
  "$PUBLIC_CORPUS_PLAN" "$EVALUATION_PLAN" "$EXTRAPOLATION_PLAN" \
  "$staging/resource_metadata.json" <<'PY'
import hashlib, json, pathlib, socket, subprocess, sys
tool, training, matrix, public, evaluation, extrapolation, output = map(
    pathlib.Path, sys.argv[1:]
)
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
metadata = {
    "protocol": "in_place_ttt_2x5090_ruler_256k_resource_run_v1",
    "hostname": socket.gethostname(),
    "tool_commit": subprocess.check_output(
        ["git", "-C", tool, "rev-parse", "HEAD"], text=True).strip(),
    "tool_clean": not subprocess.check_output(
        ["git", "-C", tool, "status", "--short"], text=True).strip(),
    "training_root": str(training.resolve()),
    "matrix_root": str(matrix.resolve()),
    "context_length": 262144,
    "training_audit_sha256": sha(training / "independent_audit.json"),
    "matrix_audit_sha256": sha(matrix / "independent_matrix_audit.json"),
    "public_corpus_plan_sha256": sha(public),
    "evaluation_plan_sha256": sha(evaluation),
    "extrapolation_plan_sha256": sha(extrapolation),
    "runner_sha256": sha(tool / "scripts/run_in_place_ttt_2x5090_ruler_256k_resource.sh"),
    "single_runner_sha256": sha(tool / "scripts/run_in_place_ttt_2x5090_ruler_single.sh"),
    "auditor_sha256": sha(tool / "scripts/audit_in_place_ttt_2x5090_ruler_256k_resource.py"),
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
    export MODEL="$model"
    export RULER_RUN_KIND=resource RULER_NUM_SAMPLES=1
    export OUTPUT_ROOT="$staging/$arm"
    bash "$TOOL_REPO/scripts/run_in_place_ttt_2x5090_ruler_single.sh" \
      262144 "$arm" "$gpu"
  ) > "$staging/${arm}.launcher.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
if [[ $failed -ne 0 ]]; then
  echo "paired RULER-256K resource boundary failed" >&2; exit 4
fi

printf 'auditing_256k_resource\n' > "$staging/state.txt"
"$PYTHON_BIN" \
  "$TOOL_REPO/scripts/audit_in_place_ttt_2x5090_ruler_256k_resource.py" \
  --root "$staging" --training-root "$TRAINING_ROOT" \
  --matrix-root "$MATRIX_ROOT" --public-corpus-plan "$PUBLIC_CORPUS_PLAN" \
  --evaluation-plan "$EVALUATION_PLAN" \
  --extrapolation-plan "$EXTRAPOLATION_PLAN" \
  --runner "$TOOL_REPO/scripts/run_in_place_ttt_2x5090_ruler_256k_resource.sh" \
  --output "$staging/independent_resource_audit.json" \
  > "$staging/independent_resource_audit.log"
printf 'completed_ruler_256k_resource\n' > "$staging/state.txt"
