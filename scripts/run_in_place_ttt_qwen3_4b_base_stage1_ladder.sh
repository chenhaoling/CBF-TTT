#!/usr/bin/env bash
set -euo pipefail

UPSTREAM_REPO=${UPSTREAM_REPO:?set the clean official checkout}
PYTHON_BIN=${PYTHON_BIN:?set the frozen training Python}
MODEL=${MODEL:?set the exact Qwen3-4B-Base directory}
MODEL_IDENTITY=${MODEL_IDENTITY:?set the passing Base identity JSON}
LEVEL_A_ROOT=${LEVEL_A_ROOT:?set the completed corrected Level-A root}
RESOURCE_INPUT_ROOT=${RESOURCE_INPUT_ROOT:?set the corrected resource input root}
OUTPUT_ROOT=${OUTPUT_ROOT:?set a new corrected Stage-1 ladder root}
PARENT_PLAN=${PARENT_PLAN:?set the parent baseline plan}
STAGE1_PLAN=${STAGE1_PLAN:?set the corrected Stage-1 resource plan}
INPUT_ADDENDUM=${INPUT_ADDENDUM:?set the corrected resource-input addendum}
TOOL_REPO=${TOOL_REPO:?set the clean tool checkout}
TOOL_COMMIT=${TOOL_COMMIT:?set the exact tool commit}
EXPECTED_HOSTNAME=${EXPECTED_HOSTNAME:-user-Super-Server}
UPSTREAM_COMMIT=be2324829b0e91c8fd10a74d4b43714fde6676e1

AUDITOR="$TOOL_REPO/scripts/audit_in_place_ttt_qwen3_4b_base_stage1_ladder.py"
RESOURCE_RUNNER="$TOOL_REPO/scripts/run_in_place_ttt_baseline_resource_pilot.sh"
DATA_MANIFEST="$RESOURCE_INPUT_ROOT/manifest.json"
DATA_AUDIT="$RESOURCE_INPUT_ROOT/independent_audit.json"
LEVEL_A_AUDIT="$LEVEL_A_ROOT/independent_audit.json"

[[ "$(hostname)" == "$EXPECTED_HOSTNAME" ]] || {
  echo "corrected Stage-1 ladder is restricted to hku-gpu2" >&2; exit 2;
}
[[ ! -e "$OUTPUT_ROOT" ]] || { echo "refusing to reuse $OUTPUT_ROOT" >&2; exit 3; }
test "$(git -C "$UPSTREAM_REPO" rev-parse HEAD)" = "$UPSTREAM_COMMIT"
test -z "$(git -C "$UPSTREAM_REPO" status --short)"
test "$(git -C "$TOOL_REPO" rev-parse HEAD)" = "$TOOL_COMMIT"
test -z "$(git -C "$TOOL_REPO" status --short)"
for path in "$MODEL/config.json" "$MODEL_IDENTITY" "$PARENT_PLAN" \
  "$STAGE1_PLAN" "$INPUT_ADDENDUM" "$DATA_MANIFEST" "$DATA_AUDIT" \
  "$LEVEL_A_AUDIT" "$AUDITOR" "$RESOURCE_RUNNER"; do test -s "$path"; done

mkdir -p "$OUTPUT_ROOT"
write_state() {
  printf '%s\n' "$1" > "$OUTPUT_ROOT/state.txt.partial"
  mv "$OUTPUT_ROOT/state.txt.partial" "$OUTPUT_ROOT/state.txt"
  date --iso-8601=seconds > "$OUTPUT_ROOT/state_time.txt.partial"
  mv "$OUTPUT_ROOT/state_time.txt.partial" "$OUTPUT_ROOT/state_time.txt"
}

write_state auditing_level_a_prerequisite
"$PYTHON_BIN" "$AUDITOR" prerequisite \
  --level-a-root "$LEVEL_A_ROOT" --level-a-audit "$LEVEL_A_AUDIT" \
  --data-manifest "$DATA_MANIFEST" --data-audit "$DATA_AUDIT" \
  --output "$OUTPUT_ROOT/level_a_prerequisite_audit.json"
"$PYTHON_BIN" "$TOOL_REPO/scripts/audit_in_place_ttt_2x5090_configs.py" \
  --output "$OUTPUT_ROOT/config_audit.json" > "$OUTPUT_ROOT/config_audit.log"

"$PYTHON_BIN" - "$OUTPUT_ROOT" "$UPSTREAM_REPO" "$TOOL_REPO" \
  "$TOOL_COMMIT" "$LEVEL_A_ROOT" "$STAGE1_PLAN" "$INPUT_ADDENDUM" \
  "$MODEL_IDENTITY" "$DATA_MANIFEST" "$DATA_AUDIT" "$AUDITOR" "$0" \
  "$PARENT_PLAN" <<'PY'
import hashlib,json,pathlib,socket,subprocess,sys
(root,upstream,tool,tool_commit,level_a,stage1_plan,input_addendum,
 identity,manifest,audit,auditor,runner,parent_plan)=sys.argv[1:]
root=pathlib.Path(root); sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
payload={
 "protocol":"in_place_ttt_qwen3_4b_base_stage1_ladder_v1",
 "hostname":socket.gethostname(),
 "upstream_commit":subprocess.check_output(["git","-C",upstream,"rev-parse","HEAD"],text=True).strip(),
 "upstream_clean":not subprocess.check_output(["git","-C",upstream,"status","--short"],text=True).strip(),
 "tool_commit":tool_commit,
 "tool_clean":not subprocess.check_output(["git","-C",tool,"status","--short"],text=True).strip(),
 "level_a_root":str(pathlib.Path(level_a).resolve()),
 "parent_plan_sha256":sha(parent_plan),
 "stage1_plan_sha256":sha(stage1_plan),"input_addendum_sha256":sha(input_addendum),
 "model_identity_sha256":sha(identity),"data_manifest_sha256":sha(manifest),
 "data_audit_sha256":sha(audit),"runner_sha256":sha(runner),
 "auditor_sha256":sha(auditor),"config_audit_sha256":sha(root/"config_audit.json"),
 "level_a_prerequisite_audit_sha256":sha(root/"level_a_prerequisite_audit.json"),
}
(root/"ladder_metadata.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY

validate_child() {
  "$PYTHON_BIN" - "$1" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]); summary=json.loads((root/"summary.json").read_text()); audit=json.loads((root/"independent_audit.json").read_text())
assert summary["completed"] and summary["activation_offload_gate_passed"] and audit["passed"]
PY
}

for length in 16384 32768; do
  data="$RESOURCE_INPUT_ROOT/resource_${length}.jsonl"
  test -s "$data"
  for arm in no_ttt ttt; do
    child="$OUTPUT_ROOT/resource_${length}_${arm}"
    config="$TOOL_REPO/configs/baseline_reproduction/qwen3_4b_2x5090_100m_stage1_${arm}.yaml"
    write_state "running_${length}_${arm}"
    date --iso-8601=seconds > "$OUTPUT_ROOT/${length}_${arm}_started_at.txt"
    env UPSTREAM_REPO="$UPSTREAM_REPO" PYTHON_BIN="$PYTHON_BIN" \
      MODEL="$MODEL" DATA="$data" DATA_MANIFEST="$DATA_MANIFEST" \
      DATA_AUDIT="$DATA_AUDIT" OUTPUT_ROOT="$child" ARM="$arm" \
      RESOURCE_PROFILE=activation_offload SEQ_LEN="$length" GLOBAL_BATCH_SIZE=2 \
      PLAN="$PARENT_PLAN" ADDENDUM="$STAGE1_PLAN" CONFIG_OVERRIDE="$config" \
      TOOL_REPO="$TOOL_REPO" EXPECTED_HOSTNAME="$EXPECTED_HOSTNAME" \
      bash "$RESOURCE_RUNNER"
    date --iso-8601=seconds > "$OUTPUT_ROOT/${length}_${arm}_completed_at.txt"
    validate_child "$child"
  done
done

write_state all_children_completed
"$PYTHON_BIN" "$AUDITOR" final --root "$OUTPUT_ROOT" \
  --level-a-root "$LEVEL_A_ROOT" --level-a-audit "$LEVEL_A_AUDIT" \
  --data-manifest "$DATA_MANIFEST" --data-audit "$DATA_AUDIT" \
  --parent-plan "$PARENT_PLAN" --stage1-plan "$STAGE1_PLAN" \
  --input-addendum "$INPUT_ADDENDUM" \
  --model-identity "$MODEL_IDENTITY" --runner "$0" \
  --output "$OUTPUT_ROOT/independent_ladder_audit.json"
"$PYTHON_BIN" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["passed"]' \
  "$OUTPUT_ROOT/independent_ladder_audit.json"
write_state completed_stage1_ladder
