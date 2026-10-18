#!/usr/bin/env bash
set -Eeuo pipefail

UPSTREAM_REPO=${UPSTREAM_REPO:?set the clean official checkout}
TOOL_REPO=${TOOL_REPO:?set the clean CBF-TTT checkout}
TOOL_COMMIT=${TOOL_COMMIT:?set the exact tool commit}
PYTHON_BIN=${PYTHON_BIN:?set the isolated OpenCompass Python}
TOKENIZER=${TOKENIZER:?set the original Qwen3-4B tokenizer directory}
TRAINING_ROOT=${TRAINING_ROOT:?set the completed paired 2x5090 formal Stage-2 root}
OUTPUT_ROOT=${OUTPUT_ROOT:?set a new RULER resource-ladder root}
PUBLIC_CORPUS_PLAN=${PUBLIC_CORPUS_PLAN:-$TOOL_REPO/IN_PLACE_TTT_BASELINE_PUBLIC_CORPUS_PLAN.md}
EVALUATION_PLAN=${EVALUATION_PLAN:-$TOOL_REPO/IN_PLACE_TTT_BASELINE_2X5090_RULER_PLAN.md}
PAPER_INFERENCE_ADDENDUM=${PAPER_INFERENCE_ADDENDUM:-$TOOL_REPO/IN_PLACE_TTT_BASELINE_PAPER_INFERENCE_CLIP_ADDENDUM.md}
EXPECTED_HOSTNAME=${EXPECTED_HOSTNAME:-user-Super-Server}
UPSTREAM_COMMIT=be2324829b0e91c8fd10a74d4b43714fde6676e1

if [[ "$(hostname)" != "$EXPECTED_HOSTNAME" ]]; then
  echo "2x5090 RULER resource ladder is restricted to hku-gpu2" >&2; exit 2
fi
if [[ -e "$OUTPUT_ROOT" || -e "${OUTPUT_ROOT}.partial" ]]; then
  echo "refusing to overwrite output or partial path" >&2; exit 3
fi
test "$(git -C "$TOOL_REPO" rev-parse HEAD)" = "$TOOL_COMMIT"
test -z "$(git -C "$TOOL_REPO" status --short)"
test "$(git -C "$UPSTREAM_REPO" rev-parse HEAD)" = "$UPSTREAM_COMMIT"
test -z "$(git -C "$UPSTREAM_REPO" status --short)"
test -s "$TRAINING_ROOT/independent_audit.json"
test -s "$PAPER_INFERENCE_ADDENDUM"
test "$(cat "$TRAINING_ROOT/state.txt")" = completed_100m_stage2

readarray -t models < <("$PYTHON_BIN" - "$TRAINING_ROOT" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1]).resolve()
audit = json.loads((root / "independent_audit.json").read_text())
if (audit.get("protocol") != "in_place_ttt_2x5090_100m_stage2_audit_v1"
        or audit.get("passed") is not True):
    raise SystemExit("paired formal Stage-2 audit did not pass")
for arm in ("no_ttt", "ttt"):
    record = audit.get("arms", {}).get(arm, {})
    model = root / arm / record.get("final_hf_relative_path", "")
    if not (model / "config.json").is_file():
        raise SystemExit(f"missing final HF checkpoint for {arm}")
    print(model)
PY
)
test ${#models[@]} -eq 2
NO_TTT_MODEL=${models[0]}
TTT_MODEL=${models[1]}

staging="${OUTPUT_ROOT}.partial"
mkdir -p "$staging"
status_file=$staging/status.txt
finish() {
  rc=$?
  trap - EXIT
  if [[ $rc -eq 0 ]]; then
    printf 'COMPLETED\nfinished_utc=%s\nexit_code=0\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$status_file"
    mv "$staging" "$OUTPUT_ROOT"
  else
    printf 'FAILED\nfinished_utc=%s\nexit_code=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" > "$status_file"
  fi
  exit "$rc"
}
trap finish EXIT
printf 'RUNNING\nstarted_utc=%s\ngpus=0,1\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$status_file"

"$PYTHON_BIN" - "$TOOL_REPO" "$TRAINING_ROOT" "$PUBLIC_CORPUS_PLAN" \
  "$EVALUATION_PLAN" "$PAPER_INFERENCE_ADDENDUM" \
  "$staging/ladder_metadata.json" <<'PY'
import hashlib, json, pathlib, socket, subprocess, sys
tool, training, public_plan, evaluation_plan, paper_addendum, output = map(
    pathlib.Path, sys.argv[1:]
)
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
metadata = {
    "protocol": "in_place_ttt_2x5090_ruler_resource_ladder_run_v1",
    "hostname": socket.gethostname(),
    "tool_commit": subprocess.check_output(
        ["git", "-C", tool, "rev-parse", "HEAD"], text=True).strip(),
    "tool_clean": not subprocess.check_output(
        ["git", "-C", tool, "status", "--short"], text=True).strip(),
    "training_root": str(training.resolve()),
    "training_audit_sha256": sha(training / "independent_audit.json"),
    "public_corpus_plan_sha256": sha(public_plan),
    "evaluation_plan_sha256": sha(evaluation_plan),
    "paper_inference_addendum_sha256": sha(paper_addendum),
    "boundaries": [4096, 32768, 65536, 131072],
    "runner_sha256": sha(
        tool / "scripts/run_in_place_ttt_2x5090_ruler_resource_ladder.sh"),
    "single_runner_sha256": sha(
        tool / "scripts/run_in_place_ttt_2x5090_ruler_single.sh"),
    "auditor_sha256": sha(
        tool / "scripts/audit_in_place_ttt_2x5090_ruler_resource_ladder.py"),
}
output.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
PY

contexts=(4096 32768 65536 131072)
suffixes=(4k 32k 64k 128k)
for index in "${!contexts[@]}"; do
  context=${contexts[$index]}
  suffix=${suffixes[$index]}
  pids=()
  for arm in no_ttt ttt; do
    if [[ "$arm" == no_ttt ]]; then model=$NO_TTT_MODEL; gpu=0
    else model=$TTT_MODEL; gpu=1; fi
    (
      export UPSTREAM_REPO TOOL_REPO TOOL_COMMIT PYTHON_BIN TOKENIZER
      export PUBLIC_CORPUS_PLAN EVALUATION_PLAN PAPER_INFERENCE_ADDENDUM TRAINING_ROOT
      export RULER_RUN_KIND=resource RULER_NUM_SAMPLES=1
      export MODEL="$model"
      export OUTPUT_ROOT="$staging/$arm/$suffix"
      bash "$TOOL_REPO/scripts/run_in_place_ttt_2x5090_ruler_single.sh" \
        "$context" "$arm" "$gpu"
    ) > "$staging/${arm}_${suffix}.launcher.log" 2>&1 &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
  done
  if [[ $failed -ne 0 ]]; then
    echo "$suffix paired RULER resource boundary failed" >&2; exit 4
  fi
done

"$PYTHON_BIN" \
  "$TOOL_REPO/scripts/audit_in_place_ttt_2x5090_ruler_resource_ladder.py" \
  --root "$staging" --public-corpus-plan "$PUBLIC_CORPUS_PLAN" \
  --evaluation-plan "$EVALUATION_PLAN" --training-root "$TRAINING_ROOT" \
  --paper-inference-addendum "$PAPER_INFERENCE_ADDENDUM" \
  --runner "$TOOL_REPO/scripts/run_in_place_ttt_2x5090_ruler_resource_ladder.sh" \
  --output "$staging/independent_resource_ladder_audit.json" \
  > "$staging/independent_resource_ladder_audit.log"
printf 'completed_ruler_resource_ladder\n' > "$staging/state.txt"
