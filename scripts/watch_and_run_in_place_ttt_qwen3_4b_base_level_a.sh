#!/usr/bin/env bash
set -euo pipefail

UPSTREAM_REPO=${UPSTREAM_REPO:?set the clean official checkout}
TRAIN_PYTHON=${TRAIN_PYTHON:?set the frozen training Python}
EVAL_PYTHON=${EVAL_PYTHON:?set the frozen OpenCompass Python}
MODEL=${MODEL:?set the exact Qwen3-4B-Base directory}
MODEL_IDENTITY=${MODEL_IDENTITY:?set the passing Base identity JSON}
CORPUS_ROOT=${CORPUS_ROOT:?set the corrected full corpus root}
CORPUS_RUN_ROOT=${CORPUS_RUN_ROOT:?set the corrected corpus wrapper run root}
CORPUS_SESSION=${CORPUS_SESSION:-public_corpus_qwen3_4b_base_full_20261019}
OUTPUT_ROOT=${OUTPUT_ROOT:?set a new corrected Level-A output root}
PARENT_PLAN=${PARENT_PLAN:?set the frozen parent baseline plan}
CORRECTION_ADDENDUM=${CORRECTION_ADDENDUM:?set the Base identity correction addendum}
EXECUTION_ADDENDUM=${EXECUTION_ADDENDUM:?set the corrected Level-A execution addendum}
RESOURCE_INPUT_ADDENDUM=${RESOURCE_INPUT_ADDENDUM:?set the resource-input addendum}
TOOL_REPO=${TOOL_REPO:?set the clean tool checkout}
TOOL_COMMIT=${TOOL_COMMIT:?set the exact tool commit}
POLL_SECONDS=${POLL_SECONDS:-60}
MAX_WAIT_SECONDS=${MAX_WAIT_SECONDS:-345600}
EXPECTED_HOSTNAME=${EXPECTED_HOSTNAME:-user-Super-Server}
REQUIRED_SOURCE_PROFILE=${REQUIRED_SOURCE_PROFILE:-full}
UPSTREAM_COMMIT=be2324829b0e91c8fd10a74d4b43714fde6676e1

BUILDER="$TOOL_REPO/scripts/build_in_place_ttt_2x5090_resource_inputs.py"
INPUT_AUDITOR="$TOOL_REPO/scripts/audit_in_place_ttt_2x5090_resource_inputs.py"
RESOURCE_RUNNER="$TOOL_REPO/scripts/run_in_place_ttt_baseline_resource_pilot.sh"
CONVERTER_RUNNER="$TOOL_REPO/scripts/convert_and_audit_in_place_ttt_baseline_checkpoints.sh"
SMOKE_RUNNER="$TOOL_REPO/scripts/run_in_place_ttt_converted_checkpoint_smokes.sh"
FINAL_AUDITOR="$TOOL_REPO/scripts/audit_in_place_ttt_qwen3_4b_base_level_a.py"
RESOURCE_INPUTS="$OUTPUT_ROOT/resource_inputs"
CORPUS_MANIFEST="$CORPUS_ROOT/manifest.json"
CORPUS_AUDIT="$CORPUS_ROOT/independent_audit.json"

if [[ "$(hostname)" != "$EXPECTED_HOSTNAME" ]]; then
  echo "corrected Base Level-A execution is restricted to hku-gpu2" >&2
  exit 2
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "refusing to reuse $OUTPUT_ROOT" >&2
  exit 3
fi
if [[ "$POLL_SECONDS" -lt 10 || "$MAX_WAIT_SECONDS" -lt "$POLL_SECONDS" ]]; then
  echo "invalid wait interval or budget" >&2
  exit 2
fi
if [[ "$REQUIRED_SOURCE_PROFILE" != full && "$REQUIRED_SOURCE_PROFILE" != preliminary_100m ]]; then
  echo "unsupported source profile: $REQUIRED_SOURCE_PROFILE" >&2
  exit 2
fi
test "$(git -C "$UPSTREAM_REPO" rev-parse HEAD)" = "$UPSTREAM_COMMIT"
test -z "$(git -C "$UPSTREAM_REPO" status --short)"
test "$(git -C "$TOOL_REPO" rev-parse HEAD)" = "$TOOL_COMMIT"
test -z "$(git -C "$TOOL_REPO" status --short)"
for path in "$TRAIN_PYTHON" "$EVAL_PYTHON" "$MODEL/config.json" \
  "$MODEL/tokenizer.json" "$MODEL_IDENTITY" "$PARENT_PLAN" \
  "$CORRECTION_ADDENDUM" "$EXECUTION_ADDENDUM" \
  "$RESOURCE_INPUT_ADDENDUM" "$BUILDER" "$INPUT_AUDITOR" \
  "$RESOURCE_RUNNER" "$CONVERTER_RUNNER" "$SMOKE_RUNNER" "$FINAL_AUDITOR"; do
  test -e "$path"
done

mkdir -p "$OUTPUT_ROOT"
CURRENT_STAGE=preflight

write_status() {
  printf '%s\n' "$1" > "$OUTPUT_ROOT/status.txt.partial"
  mv "$OUTPUT_ROOT/status.txt.partial" "$OUTPUT_ROOT/status.txt"
  date --iso-8601=seconds > "$OUTPUT_ROOT/status_time.txt.partial"
  mv "$OUTPUT_ROOT/status_time.txt.partial" "$OUTPUT_ROOT/status_time.txt"
}

fail_current_stage() {
  local code=${1:-1}
  trap - ERR
  printf 'stage=%s exit=%s\n' "$CURRENT_STAGE" "$code" >> "$OUTPUT_ROOT/error.log"
  write_status "FAILED_${CURRENT_STAGE}"
  exit "$code"
}
on_error() {
  fail_current_stage "$?"
}
trap on_error ERR

write_status PREFLIGHT
sha256sum "$0" "$PARENT_PLAN" "$CORRECTION_ADDENDUM" \
  "$EXECUTION_ADDENDUM" "$RESOURCE_INPUT_ADDENDUM" "$BUILDER" \
  "$INPUT_AUDITOR" "$RESOURCE_RUNNER" "$CONVERTER_RUNNER" \
  "$SMOKE_RUNNER" "$FINAL_AUDITOR" > "$OUTPUT_ROOT/static_inputs.sha256"
printf '%s\n' "$TOOL_COMMIT" > "$OUTPUT_ROOT/tool_commit.txt"

"$TRAIN_PYTHON" - "$MODEL" "$MODEL_IDENTITY" <<'PY'
import hashlib, json, pathlib, sys
model, identity_path = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
identity = json.loads(identity_path.read_text())
assert identity["passed"] is True and all(identity["checks"].values())
assert identity["repository"] == "Qwen/Qwen3-4B-Base"
assert identity["revision"] == "906bfd4b4dc7f14ee4320094d8b41684abff8539"
for name in ("config.json", "tokenizer.json", "tokenizer_config.json",
             "model.safetensors.index.json"):
    path, saved = model / name, identity["files"][name]
    assert path.stat().st_size == saved["size"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == saved["sha256"]
config = json.loads((model / "config.json").read_text())
assert config["architectures"] == ["Qwen3ForCausalLM"]
assert config["max_position_embeddings"] == 32768
assert config["eos_token_id"] == 151643
PY

CURRENT_STAGE=waiting_base_corpus
started=$(date +%s)
while true; do
  write_status WAITING_BASE_CORPUS
  corpus_status=$(cat "$CORPUS_RUN_ROOT/status.txt" 2>/dev/null || true)
  if [[ "$corpus_status" == COMPLETE ]]; then
    break
  fi
  if [[ "$corpus_status" == FAILED_* ]]; then
    echo "corrected full corpus failed: $corpus_status" >&2
    false
  fi
  if ! tmux has-session -t "$CORPUS_SESSION" 2>/dev/null; then
    sleep 5
    corpus_status=$(cat "$CORPUS_RUN_ROOT/status.txt" 2>/dev/null || true)
    [[ "$corpus_status" == COMPLETE ]] || {
      echo "corrected corpus handle disappeared in state $corpus_status" >&2
      false
    }
    break
  fi
  now=$(date +%s)
  if [[ $((now - started)) -ge "$MAX_WAIT_SECONDS" ]]; then
    echo "timed out waiting for corrected full corpus" >&2
    false
  fi
  sleep "$POLL_SECONDS"
done

CURRENT_STAGE=corpus_release_audit
write_status VERIFYING_BASE_CORPUS
"$TRAIN_PYTHON" - "$CORPUS_ROOT" "$MODEL" "$MODEL_IDENTITY" \
  "$REQUIRED_SOURCE_PROFILE" \
  "$OUTPUT_ROOT/corpus_release.json" <<'PY'
import hashlib, json, pathlib, sys
root, model, identity_path = map(pathlib.Path, sys.argv[1:4])
profile, output = sys.argv[4], pathlib.Path(sys.argv[5])
manifest_path, audit_path = root / "manifest.json", root / "independent_audit.json"
assert manifest_path.is_file() and audit_path.is_file()
assert not root.with_name(root.name + ".partial").exists()
manifest, audit = json.loads(manifest_path.read_text()), json.loads(audit_path.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
expected = {
    "preliminary_100m": {"records": 2048, "synthetic_eos_tokens": 0,
                         "training_tokens": 98566144},
    "full": {"records": 724800, "synthetic_eos_tokens": 0,
             "training_tokens": 34999369728},
}
assert profile in expected and manifest["profile"] == profile
assert {key: manifest["totals"][key] for key in expected[profile]} == expected[profile]
assert len(manifest["totals"]["combined_curriculum_sha256"]) == 64
assert audit["passed"] is True and all(audit["checks"].values())
assert audit["bindings"]["manifest_sha256"] == sha(manifest_path)
identity = json.loads(identity_path.read_text())
assets = manifest["bindings"]["tokenizer_assets"]
for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
    assert assets[name]["sha256"] == identity["files"][name]["sha256"]
payload = {"protocol": "qwen3_4b_base_corpus_release_v2", "profile": profile,
           "manifest_sha256": sha(manifest_path), "audit_sha256": sha(audit_path),
           "model_identity_sha256": sha(identity_path), "passed": True}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

corpus_manifest_sha=$("$TRAIN_PYTHON" - "$OUTPUT_ROOT/corpus_release.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))["manifest_sha256"])
PY
)
sha256sum "$CORPUS_MANIFEST" "$CORPUS_AUDIT" "$MODEL_IDENTITY" \
  > "$OUTPUT_ROOT/released_inputs.sha256"

CURRENT_STAGE=resource_input_build
write_status BUILDING_RESOURCE_INPUTS
env CUDA_VISIBLE_DEVICES= "$TRAIN_PYTHON" "$BUILDER" \
  --source-root "$CORPUS_ROOT" --source-audit "$CORPUS_AUDIT" \
  --tokenizer "$MODEL" --output-root "$RESOURCE_INPUTS" \
  --plan "$RESOURCE_INPUT_ADDENDUM" \
  --expected-source-manifest-sha256 "$corpus_manifest_sha" \
  --required-source-profile "$REQUIRED_SOURCE_PROFILE" > "$OUTPUT_ROOT/resource_input_build.log"

CURRENT_STAGE=resource_input_audit
write_status AUDITING_RESOURCE_INPUTS
env CUDA_VISIBLE_DEVICES= "$TRAIN_PYTHON" "$INPUT_AUDITOR" \
  --root "$RESOURCE_INPUTS" --source-root "$CORPUS_ROOT" \
  --source-audit "$CORPUS_AUDIT" --tokenizer "$MODEL" \
  --plan "$RESOURCE_INPUT_ADDENDUM" --builder "$BUILDER" \
  --expected-source-manifest-sha256 "$corpus_manifest_sha" \
  --required-source-profile "$REQUIRED_SOURCE_PROFILE" \
  --output "$RESOURCE_INPUTS/independent_audit.json" \
  > "$OUTPUT_ROOT/resource_input_audit.log"
"$TRAIN_PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["passed"] and all(d["checks"].values())' \
  "$RESOURCE_INPUTS/independent_audit.json"

"$TRAIN_PYTHON" "$TOOL_REPO/scripts/audit_in_place_ttt_2x5090_configs.py" \
  --output "$OUTPUT_ROOT/config_audit.json" > "$OUTPUT_ROOT/config_audit.log"

run_arm() {
  local arm=$1
  local config="$TOOL_REPO/configs/baseline_reproduction/qwen3_4b_2x5090_100m_stage1_${arm}.yaml"
  date --iso-8601=seconds > "$OUTPUT_ROOT/${arm}_started_at.txt"
  local code
  if env \
    UPSTREAM_REPO="$UPSTREAM_REPO" PYTHON_BIN="$TRAIN_PYTHON" \
    MODEL="$MODEL" DATA="$RESOURCE_INPUTS/resource_8192.jsonl" \
    OUTPUT_ROOT="$OUTPUT_ROOT/resource_8k_${arm}" ARM="$arm" \
    RESOURCE_PROFILE=activation_offload SEQ_LEN=8192 GLOBAL_BATCH_SIZE=2 \
    PLAN="$PARENT_PLAN" ADDENDUM="$EXECUTION_ADDENDUM" \
    CONFIG_OVERRIDE="$config" TOOL_REPO="$TOOL_REPO" \
    DATA_MANIFEST="$RESOURCE_INPUTS/manifest.json" \
    DATA_AUDIT="$RESOURCE_INPUTS/independent_audit.json" \
    EXPECTED_HOSTNAME="$EXPECTED_HOSTNAME" \
    bash "$RESOURCE_RUNNER"; then
    :
  else
    code=$?
    return "$code"
  fi
  date --iso-8601=seconds > "$OUTPUT_ROOT/${arm}_completed_at.txt"
  if "$TRAIN_PYTHON" - "$OUTPUT_ROOT/resource_8k_${arm}" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1])
summary=json.loads((root/"summary.json").read_text())
audit=json.loads((root/"independent_audit.json").read_text())
assert summary["completed"] and summary["activation_offload_gate_passed"]
assert audit["passed"]
PY
  then
    :
  else
    code=$?
    return "$code"
  fi
}

CURRENT_STAGE=resource_no_ttt
write_status RUNNING_RESOURCE_NO_TTT
if run_arm no_ttt; then
  :
else
  code=$?
  fail_current_stage "$code"
fi
CURRENT_STAGE=resource_ttt
write_status RUNNING_RESOURCE_TTT
if run_arm ttt; then
  :
else
  code=$?
  fail_current_stage "$code"
fi
printf '%s\n' completed_both_arms > "$OUTPUT_ROOT/state.txt"

CURRENT_STAGE=conversion
write_status CONVERTING_CHECKPOINTS
env UPSTREAM_REPO="$UPSTREAM_REPO" PYTHON_BIN="$TRAIN_PYTHON" \
  OUTPUT_ROOT="$OUTPUT_ROOT" bash "$CONVERTER_RUNNER"

CURRENT_STAGE=ruler_smoke
write_status RUNNING_RULER_SMOKE
env UPSTREAM_REPO="$UPSTREAM_REPO" EVAL_PYTHON="$EVAL_PYTHON" \
  TOKENIZER="$MODEL" OUTPUT_ROOT="$OUTPUT_ROOT" PLAN="$PARENT_PLAN" \
  TOOL_REPO="$TOOL_REPO" TOKENIZER_AUDIT_ADDENDUM="$CORRECTION_ADDENDUM" \
  SMOKE_SUFFIX=qwen3_4b_base_corrected_smoke \
  REQUIRE_ZERO_TOKENIZER_REGEX_WARNINGS=1 bash "$SMOKE_RUNNER"

CURRENT_STAGE=final_audit
write_status RUNNING_FINAL_AUDIT
"$TRAIN_PYTHON" "$FINAL_AUDITOR" --root "$OUTPUT_ROOT" \
  --model "$MODEL" --identity "$MODEL_IDENTITY" \
  --corpus-root "$CORPUS_ROOT" --corpus-audit "$CORPUS_AUDIT" \
  --resource-inputs "$RESOURCE_INPUTS" \
  --resource-audit "$RESOURCE_INPUTS/independent_audit.json" \
  --parent-plan "$PARENT_PLAN" --correction-addendum "$CORRECTION_ADDENDUM" \
  --execution-addendum "$EXECUTION_ADDENDUM" \
  --resource-input-addendum "$RESOURCE_INPUT_ADDENDUM" \
  --required-source-profile "$REQUIRED_SOURCE_PROFILE" \
  --output "$OUTPUT_ROOT/independent_audit.json" \
  > "$OUTPUT_ROOT/final_audit.log"
"$TRAIN_PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["passed"] and all(d["checks"].values())' \
  "$OUTPUT_ROOT/independent_audit.json"

sha256sum "$OUTPUT_ROOT/independent_audit.json" \
  "$OUTPUT_ROOT/resource_8k_no_ttt/summary.json" \
  "$OUTPUT_ROOT/resource_8k_ttt/summary.json" \
  "$OUTPUT_ROOT/ruler_4k_no_ttt_step1_qwen3_4b_base_corrected_smoke/summary.json" \
  "$OUTPUT_ROOT/ruler_4k_ttt_step1_qwen3_4b_base_corrected_smoke/summary.json" \
  > "$OUTPUT_ROOT/final_outputs.sha256"
CURRENT_STAGE=complete
write_status COMPLETE
trap - ERR
