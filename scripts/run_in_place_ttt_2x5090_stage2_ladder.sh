#!/usr/bin/env bash
set -euo pipefail

UPSTREAM_REPO=${UPSTREAM_REPO:?set the clean official checkout}
PYTHON_BIN=${PYTHON_BIN:?set the baseline training Python}
STAGE1_ROOT=${STAGE1_ROOT:?set the completed paired formal Stage-1 root}
RESOURCE_INPUT_ROOT=${RESOURCE_INPUT_ROOT:?set the audited exact input root}
OUTPUT_ROOT=${OUTPUT_ROOT:?set a new Stage-2 ladder output root}
BASELINE_PLAN=${BASELINE_PLAN:?set the parent baseline plan}
RESOURCE_PLAN=${RESOURCE_PLAN:?set the frozen Stage-2 resource plan}
EXECUTION_ADDENDUM=${EXECUTION_ADDENDUM:?set the frozen 2x5090 execution addendum}
INPUT_ADDENDUM=${INPUT_ADDENDUM:?set the exact-input addendum}
TOOL_REPO=${TOOL_REPO:?set the clean CBF-TTT checkout}
TOOL_COMMIT=${TOOL_COMMIT:?set the exact tool commit}
EXPECTED_HOSTNAME=${EXPECTED_HOSTNAME:-user-Super-Server}
UPSTREAM_COMMIT=be2324829b0e91c8fd10a74d4b43714fde6676e1
DATA_MANIFEST=$RESOURCE_INPUT_ROOT/manifest.json
DATA_AUDIT=$RESOURCE_INPUT_ROOT/independent_audit.json
STAGE1_AUDIT=$STAGE1_ROOT/independent_audit.json
NO_TTT_CONFIG=$TOOL_REPO/configs/baseline_reproduction/qwen3_4b_2x5090_100m_stage2_no_ttt.yaml
TTT_CONFIG=$TOOL_REPO/configs/baseline_reproduction/qwen3_4b_2x5090_100m_stage2_ttt.yaml

if [[ "$(hostname)" != "$EXPECTED_HOSTNAME" ]]; then
  echo "the Stage-2 resource ladder is restricted to hku-gpu2" >&2; exit 2
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "refusing to reuse Stage-2 ladder output" >&2; exit 3
fi
test "$(git -C "$UPSTREAM_REPO" rev-parse HEAD)" = "$UPSTREAM_COMMIT"
test -z "$(git -C "$UPSTREAM_REPO" status --short)"
test "$(git -C "$TOOL_REPO" rev-parse HEAD)" = "$TOOL_COMMIT"
test -z "$(git -C "$TOOL_REPO" status --short)"
test -s "$STAGE1_AUDIT"; test "$(cat "$STAGE1_ROOT/state.txt")" = completed_100m_stage1
test -s "$DATA_MANIFEST"; test -s "$DATA_AUDIT"
test -s "$BASELINE_PLAN"; test -s "$RESOURCE_PLAN"; test -s "$EXECUTION_ADDENDUM"
test -s "$INPUT_ADDENDUM"; test -s "$NO_TTT_CONFIG"; test -s "$TTT_CONFIG"

mkdir -p "$OUTPUT_ROOT"

# Rehash both complete Stage-1 HF trees before any resource child can touch CUDA.
"$PYTHON_BIN" - "$TOOL_REPO" "$STAGE1_ROOT" "$STAGE1_AUDIT" \
  "$OUTPUT_ROOT/start_models.json" <<'PY'
import json, pathlib, sys
tool, stage1_root, audit_path, output = map(pathlib.Path, sys.argv[1:])
sys.path.insert(0, str(tool))
from scripts.audit_in_place_ttt_2x5090_stage2_ladder import validate_start_model
audit = json.loads(audit_path.read_text())
if (audit.get("protocol") != "in_place_ttt_2x5090_100m_stage1_audit_v1"
        or audit.get("passed") is not True):
    raise SystemExit("paired formal Stage-1 audit did not pass")
records, errors = {}, []
for arm in ("no_ttt", "ttt"):
    arm_errors, record = validate_start_model(stage1_root, audit, arm)
    errors.extend(arm_errors); records[arm] = record
if errors:
    raise SystemExit("; ".join(errors))
output.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
PY

"$PYTHON_BIN" "$TOOL_REPO/scripts/audit_in_place_ttt_2x5090_configs.py" \
  --output "$OUTPUT_ROOT/config_audit.json" > "$OUTPUT_ROOT/config_audit.stdout"

"$PYTHON_BIN" - "$UPSTREAM_REPO" "$TOOL_REPO" "$TOOL_COMMIT" \
  "$STAGE1_ROOT" "$STAGE1_AUDIT" "$RESOURCE_PLAN" "$EXECUTION_ADDENDUM" \
  "$INPUT_ADDENDUM" "$DATA_MANIFEST" "$DATA_AUDIT" \
  "$NO_TTT_CONFIG" "$TTT_CONFIG" "$OUTPUT_ROOT/config_audit.json" \
  "$OUTPUT_ROOT/start_models.json" "$OUTPUT_ROOT/ladder_metadata.json" <<'PY'
import hashlib, json, pathlib, socket, subprocess, sys
(upstream, tool, tool_commit, stage1_root, stage1_audit, resource_plan,
 execution_addendum, input_addendum, manifest, data_audit, no_ttt_config,
 ttt_config, config_audit, start_models, output) = sys.argv[1:]
sha = lambda path: hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
metadata = {
    "protocol": "in_place_ttt_2x5090_stage2_ladder_v1",
    "hostname": socket.gethostname(),
    "upstream_commit": subprocess.check_output(
        ["git", "-C", upstream, "rev-parse", "HEAD"], text=True).strip(),
    "upstream_clean": not subprocess.check_output(
        ["git", "-C", upstream, "status", "--short"], text=True).strip(),
    "tool_commit": tool_commit,
    "tool_clean": not subprocess.check_output(
        ["git", "-C", tool, "status", "--short"], text=True).strip(),
    "stage1_root": str(pathlib.Path(stage1_root).resolve()),
    "stage1_audit_sha256": sha(stage1_audit),
    "resource_plan_sha256": sha(resource_plan),
    "execution_addendum_sha256": sha(execution_addendum),
    "input_addendum_sha256": sha(input_addendum),
    "data_manifest_sha256": sha(manifest),
    "data_audit_sha256": sha(data_audit),
    "no_ttt_config_sha256": sha(no_ttt_config),
    "ttt_config_sha256": sha(ttt_config),
    "config_audit_sha256": sha(config_audit),
    "start_models_sha256": sha(start_models),
    "runner_sha256": sha(pathlib.Path(tool) / "scripts/run_in_place_ttt_2x5090_stage2_ladder.sh"),
    "auditor_sha256": sha(pathlib.Path(tool) / "scripts/audit_in_place_ttt_2x5090_stage2_ladder.py"),
}
pathlib.Path(output).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
PY

write_state() {
  printf '%s\n' "$1" > "$OUTPUT_ROOT/state.txt.partial"
  mv "$OUTPUT_ROOT/state.txt.partial" "$OUTPUT_ROOT/state.txt"
  date --iso-8601=seconds > "$OUTPUT_ROOT/state_time.txt.partial"
  mv "$OUTPUT_ROOT/state_time.txt.partial" "$OUTPUT_ROOT/state_time.txt"
}

field() {
  "$PYTHON_BIN" - "$OUTPUT_ROOT/start_models.json" "$1" <<'PY'
import json, pathlib, sys
print(json.loads(pathlib.Path(sys.argv[1]).read_text())[sys.argv[2]]["root"])
PY
}

validate_child() {
  "$PYTHON_BIN" - "$1" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
summary = json.loads((root / "summary.json").read_text())
audit = json.loads((root / "independent_audit.json").read_text())
if (summary.get("completed") is not True
        or summary.get("activation_offload_gate_passed") is not True
        or audit.get("passed") is not True):
    raise SystemExit(1)
PY
}

for length in 65536 131072; do
  data=$RESOURCE_INPUT_ROOT/resource_${length}.jsonl
  test -s "$data"
  for arm in no_ttt ttt; do
    child=$OUTPUT_ROOT/resource_${length}_${arm}
    model=$(field "$arm")
    if [[ "$arm" == no_ttt ]]; then config=$NO_TTT_CONFIG; else config=$TTT_CONFIG; fi
    write_state "running_${length}_${arm}"
    set +e
    env \
      UPSTREAM_REPO="$UPSTREAM_REPO" PYTHON_BIN="$PYTHON_BIN" \
      MODEL="$model" DATA="$data" DATA_MANIFEST="$DATA_MANIFEST" \
      DATA_AUDIT="$DATA_AUDIT" OUTPUT_ROOT="$child" ARM="$arm" \
      RESOURCE_PROFILE=activation_offload SEQ_LEN="$length" GLOBAL_BATCH_SIZE=2 \
      PLAN="$BASELINE_PLAN" ADDENDUM="$EXECUTION_ADDENDUM" \
      CONFIG_OVERRIDE="$config" TOOL_REPO="$TOOL_REPO" \
      EXPECTED_HOSTNAME="$EXPECTED_HOSTNAME" \
      bash "$TOOL_REPO/scripts/run_in_place_ttt_baseline_resource_pilot.sh"
    status=$?
    set -e
    printf '%s\n' "$status" > "$OUTPUT_ROOT/${length}_${arm}_runner_exit_code.txt"
    if [[ "$status" -ne 0 ]]; then
      write_state "failed_${length}_${arm}"
      exit 4
    fi
    if ! validate_child "$child"; then
      write_state "invalid_${length}_${arm}"
      exit 5
    fi
  done
done

write_state all_children_completed
set +e
"$PYTHON_BIN" "$TOOL_REPO/scripts/audit_in_place_ttt_2x5090_stage2_ladder.py" \
  --root "$OUTPUT_ROOT" --stage1-root "$STAGE1_ROOT" \
  --resource-plan "$RESOURCE_PLAN" --execution-addendum "$EXECUTION_ADDENDUM" \
  --input-addendum "$INPUT_ADDENDUM" --data-manifest "$DATA_MANIFEST" \
  --data-audit "$DATA_AUDIT" --no-ttt-config "$NO_TTT_CONFIG" \
  --ttt-config "$TTT_CONFIG" \
  --runner "$TOOL_REPO/scripts/run_in_place_ttt_2x5090_stage2_ladder.sh" \
  --output "$OUTPUT_ROOT/independent_ladder_audit.json"
audit_status=$?
set -e
if [[ "$audit_status" -ne 0 ]]; then
  write_state failed_independent_stage2_ladder_audit
  exit 6
fi
write_state completed_stage2_ladder
