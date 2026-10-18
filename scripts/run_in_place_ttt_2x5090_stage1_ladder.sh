#!/usr/bin/env bash
set -euo pipefail

UPSTREAM_REPO=${UPSTREAM_REPO:?set the clean official checkout}
PYTHON_BIN=${PYTHON_BIN:?set the baseline training Python}
MODEL=${MODEL:?set the Qwen3-4B model directory}
EIGHT_K_ROOT=${EIGHT_K_ROOT:?set the completed matched 8K watcher root}
RESOURCE_INPUT_ROOT=${RESOURCE_INPUT_ROOT:?set the audited exact input root}
OUTPUT_ROOT=${OUTPUT_ROOT:?set a new Stage-1 ladder output root}
PLAN=${PLAN:?set the parent baseline plan}
ADDENDUM=${ADDENDUM:?set the frozen 2x5090 execution addendum}
INPUT_ADDENDUM=${INPUT_ADDENDUM:?set the exact-input addendum}
TOOL_REPO=${TOOL_REPO:?set the clean CBF-TTT checkout}
TOOL_COMMIT=${TOOL_COMMIT:?set the exact tool commit}
EXPECTED_HOSTNAME=${EXPECTED_HOSTNAME:-user-Super-Server}
UPSTREAM_COMMIT=be2324829b0e91c8fd10a74d4b43714fde6676e1
DATA_MANIFEST=$RESOURCE_INPUT_ROOT/manifest.json
DATA_AUDIT=$RESOURCE_INPUT_ROOT/independent_audit.json

if [[ "$(hostname)" != "$EXPECTED_HOSTNAME" ]]; then
  echo "the Stage-1 ladder is restricted to hku-gpu2" >&2
  exit 2
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "refusing to reuse $OUTPUT_ROOT" >&2
  exit 3
fi
test "$(git -C "$UPSTREAM_REPO" rev-parse HEAD)" = "$UPSTREAM_COMMIT"
test -z "$(git -C "$UPSTREAM_REPO" status --short)"
test "$(git -C "$TOOL_REPO" rev-parse HEAD)" = "$TOOL_COMMIT"
test -z "$(git -C "$TOOL_REPO" status --short)"
test -s "$MODEL/config.json"; test -s "$PLAN"; test -s "$ADDENDUM"
test -s "$INPUT_ADDENDUM"; test -s "$DATA_MANIFEST"; test -s "$DATA_AUDIT"

mkdir -p "$OUTPUT_ROOT"
write_state() {
  printf '%s\n' "$1" > "$OUTPUT_ROOT/state.txt.partial"
  mv "$OUTPUT_ROOT/state.txt.partial" "$OUTPUT_ROOT/state.txt"
  date --iso-8601=seconds > "$OUTPUT_ROOT/state_time.txt.partial"
  mv "$OUTPUT_ROOT/state_time.txt.partial" "$OUTPUT_ROOT/state_time.txt"
}

write_state auditing_8k_prerequisite
"$PYTHON_BIN" "$TOOL_REPO/scripts/audit_in_place_ttt_2x5090_stage1_ladder.py" \
  prerequisite --eight-k-root "$EIGHT_K_ROOT" \
  --output "$OUTPUT_ROOT/eight_k_prerequisite_audit.json"
"$PYTHON_BIN" "$TOOL_REPO/scripts/audit_in_place_ttt_2x5090_configs.py" \
  --output "$OUTPUT_ROOT/config_audit.json" > "$OUTPUT_ROOT/config_audit.stdout"

"$PYTHON_BIN" - "$OUTPUT_ROOT" "$UPSTREAM_REPO" "$TOOL_REPO" "$TOOL_COMMIT" \
  "$PLAN" "$ADDENDUM" "$INPUT_ADDENDUM" "$DATA_MANIFEST" "$DATA_AUDIT" \
  "$EIGHT_K_ROOT" <<'PY'
import hashlib, json, pathlib, socket, subprocess, sys
(root, upstream, tool, tool_commit, plan, addendum, input_addendum,
 manifest, data_audit, eight_k) = sys.argv[1:]
root = pathlib.Path(root)
sha = lambda path: hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
metadata = {
    "protocol": "in_place_ttt_2x5090_stage1_ladder_v1",
    "hostname": socket.gethostname(),
    "upstream_commit": subprocess.check_output(
        ["git", "-C", upstream, "rev-parse", "HEAD"], text=True).strip(),
    "upstream_clean": not subprocess.check_output(
        ["git", "-C", upstream, "status", "--short"], text=True).strip(),
    "tool_commit": tool_commit,
    "tool_clean": not subprocess.check_output(
        ["git", "-C", tool, "status", "--short"], text=True).strip(),
    "eight_k_root": str(pathlib.Path(eight_k).resolve()),
    "execution_addendum_sha256": sha(addendum),
    "input_addendum_sha256": sha(input_addendum),
    "data_manifest_sha256": sha(manifest),
    "data_audit_sha256": sha(data_audit),
    "config_audit_sha256": sha(root / "config_audit.json"),
    "eight_k_prerequisite_audit_sha256": sha(root / "eight_k_prerequisite_audit.json"),
    "runner_sha256": sha(pathlib.Path(tool) / "scripts/run_in_place_ttt_2x5090_stage1_ladder.sh"),
    "auditor_sha256": sha(pathlib.Path(tool) / "scripts/audit_in_place_ttt_2x5090_stage1_ladder.py"),
}
(root / "ladder_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
PY

validate_child() {
  local child=$1
  "$PYTHON_BIN" - "$child" <<'PY'
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

for length in 16384 32768; do
  data=$RESOURCE_INPUT_ROOT/resource_${length}.jsonl
  test -s "$data"
  for arm in no_ttt ttt; do
    child=$OUTPUT_ROOT/resource_${length}_${arm}
    config=$TOOL_REPO/configs/baseline_reproduction/qwen3_4b_2x5090_100m_stage1_${arm}.yaml
    write_state "running_${length}_${arm}"
    set +e
    env \
      UPSTREAM_REPO="$UPSTREAM_REPO" PYTHON_BIN="$PYTHON_BIN" \
      MODEL="$MODEL" DATA="$data" DATA_MANIFEST="$DATA_MANIFEST" \
      DATA_AUDIT="$DATA_AUDIT" OUTPUT_ROOT="$child" ARM="$arm" \
      RESOURCE_PROFILE=activation_offload SEQ_LEN="$length" GLOBAL_BATCH_SIZE=2 \
      PLAN="$PLAN" ADDENDUM="$ADDENDUM" CONFIG_OVERRIDE="$config" \
      TOOL_REPO="$TOOL_REPO" EXPECTED_HOSTNAME="$EXPECTED_HOSTNAME" \
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
"$PYTHON_BIN" "$TOOL_REPO/scripts/audit_in_place_ttt_2x5090_stage1_ladder.py" \
  final --root "$OUTPUT_ROOT" --eight-k-root "$EIGHT_K_ROOT" \
  --data-manifest "$DATA_MANIFEST" --data-audit "$DATA_AUDIT" \
  --execution-addendum "$ADDENDUM" --input-addendum "$INPUT_ADDENDUM" \
  --runner "$TOOL_REPO/scripts/run_in_place_ttt_2x5090_stage1_ladder.sh" \
  --output "$OUTPUT_ROOT/independent_ladder_audit.json"
audit_status=$?
set -e
if [[ "$audit_status" -ne 0 ]]; then
  write_state failed_independent_ladder_audit
  exit 6
fi
write_state completed_stage1_ladder
