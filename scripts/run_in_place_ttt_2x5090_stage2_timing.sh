#!/usr/bin/env bash
set -Eeuo pipefail

UPSTREAM_REPO=${UPSTREAM_REPO:?set the clean official checkout}
TOOL_REPO=${TOOL_REPO:?set the clean CBF-TTT checkout}
TOOL_COMMIT=${TOOL_COMMIT:?set the exact tool commit}
PYTHON_BIN=${PYTHON_BIN:?set the official training Python}
CORPUS_ROOT=${CORPUS_ROOT:?set the independently audited full corpus}
STAGE1_ROOT=${STAGE1_ROOT:?set the completed paired formal Stage-1 root}
STAGE2_LADDER_ROOT=${STAGE2_LADDER_ROOT:?set the completed Stage-2 resource ladder}
OUTPUT_ROOT=${OUTPUT_ROOT:?set a new Stage-2 timing output root}
TIMING_PLAN=${TIMING_PLAN:?set the frozen Stage-2 timing plan}
EXECUTION_ADDENDUM=${EXECUTION_ADDENDUM:?set the frozen 2x5090 addendum}
EXPECTED_HOSTNAME=${EXPECTED_HOSTNAME:-user-Super-Server}
UPSTREAM_COMMIT=be2324829b0e91c8fd10a74d4b43714fde6676e1
STEPS=3
SEQ_LEN=131072
GLOBAL_BATCH_SIZE=64
TRAIN_SIZE=25165824
DATA=$CORPUS_ROOT/stage2_128k.jsonl
CORPUS_MANIFEST=$CORPUS_ROOT/manifest.json
CORPUS_AUDIT=$CORPUS_ROOT/independent_audit.json
STAGE1_AUDIT=$STAGE1_ROOT/independent_audit.json
STAGE2_LADDER_AUDIT=$STAGE2_LADDER_ROOT/independent_ladder_audit.json
NO_TTT_CONFIG=$TOOL_REPO/configs/baseline_reproduction/qwen3_4b_2x5090_100m_stage2_no_ttt.yaml
TTT_CONFIG=$TOOL_REPO/configs/baseline_reproduction/qwen3_4b_2x5090_100m_stage2_ttt.yaml

if [[ "$(hostname)" != "$EXPECTED_HOSTNAME" ]]; then
  echo "the Stage-2 timing pilot is restricted to hku-gpu2" >&2; exit 2
fi
if [[ -e "$OUTPUT_ROOT" || -e "${OUTPUT_ROOT}.partial" ]]; then
  echo "refusing to overwrite Stage-2 timing output" >&2; exit 3
fi
test "$(git -C "$UPSTREAM_REPO" rev-parse HEAD)" = "$UPSTREAM_COMMIT"
test -z "$(git -C "$UPSTREAM_REPO" status --short)"
test "$(git -C "$TOOL_REPO" rev-parse HEAD)" = "$TOOL_COMMIT"
test -z "$(git -C "$TOOL_REPO" status --short)"
test -s "$DATA"; test -s "$CORPUS_MANIFEST"; test -s "$CORPUS_AUDIT"
test -s "$STAGE1_AUDIT"; test -s "$STAGE2_LADDER_AUDIT"
test "$(cat "$STAGE1_ROOT/state.txt")" = completed_100m_stage1
test "$(cat "$STAGE2_LADDER_ROOT/state.txt")" = completed_stage2_ladder
test -s "$TIMING_PLAN"; test -s "$EXECUTION_ADDENDUM"
test -s "$NO_TTT_CONFIG"; test -s "$TTT_CONFIG"

staging=${OUTPUT_ROOT}.partial
mkdir -p "$staging"

# Rehash both arm-specific Stage-1 starts and bind them to the completed
# Stage-2 ladder before any CUDA preflight.
"$PYTHON_BIN" - "$TOOL_REPO" "$STAGE1_ROOT" "$STAGE1_AUDIT" \
  "$STAGE2_LADDER_AUDIT" "$staging/start_models.json" <<'PY'
import json, pathlib, sys
tool, stage1_root, stage1_audit_path, ladder_path, output = map(pathlib.Path, sys.argv[1:])
sys.path.insert(0, str(tool))
from scripts.audit_in_place_ttt_2x5090_stage2_ladder import validate_start_model
stage1 = json.loads(stage1_audit_path.read_text())
ladder = json.loads(ladder_path.read_text())
if stage1.get("passed") is not True or ladder.get("passed") is not True:
    raise SystemExit("formal Stage-1 or Stage-2 ladder audit did not pass")
records, errors = {}, []
for arm in ("no_ttt", "ttt"):
    arm_errors, record = validate_start_model(stage1_root, stage1, arm)
    errors.extend(arm_errors); records[arm] = record
if errors or ladder.get("start_models") != records:
    raise SystemExit("arm-specific Stage-1 model binding differs from ladder")
output.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
PY

DATA_SHA=$("$PYTHON_BIN" - "$DATA" "$CORPUS_MANIFEST" "$CORPUS_AUDIT" <<'PY'
import hashlib, json, pathlib, sys
data, manifest_path, audit_path = map(pathlib.Path, sys.argv[1:])
def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20): digest.update(block)
    return digest.hexdigest()
manifest, audit = json.loads(manifest_path.read_text()), json.loads(audit_path.read_text())
stage = next((item for item in manifest.get("stages", [])
              if item.get("name") == "stage2_128k"), {})
artifact = stage.get("artifacts", {}).get(data.name, {})
data_sha = sha(data)
if (manifest.get("protocol") != "in_place_ttt_public_corpus_v1"
        or manifest.get("profile") != "full" or audit.get("passed") is not True
        or audit.get("profile") != "full"
        or audit.get("bindings", {}).get("manifest_sha256") != sha(manifest_path)
        or artifact.get("sha256") != data_sha
        or artifact.get("size_bytes") != data.stat().st_size):
    raise SystemExit("full Stage-2 corpus binding is invalid")
print(data_sha)
PY
)

"$PYTHON_BIN" "$TOOL_REPO/scripts/audit_in_place_ttt_2x5090_configs.py" \
  --output "$staging/config_audit.json" > "$staging/config_audit.stdout"

"$PYTHON_BIN" - "$UPSTREAM_REPO" "$TOOL_REPO" "$STAGE1_ROOT" \
  "$STAGE2_LADDER_ROOT" "$STAGE1_AUDIT" "$STAGE2_LADDER_AUDIT" \
  "$TIMING_PLAN" "$EXECUTION_ADDENDUM" "$CORPUS_MANIFEST" "$CORPUS_AUDIT" \
  "$DATA_SHA" "$NO_TTT_CONFIG" "$TTT_CONFIG" "$staging/config_audit.json" \
  "$staging/start_models.json" "$staging/run_metadata.json" \
  "$STEPS" "$SEQ_LEN" "$GLOBAL_BATCH_SIZE" "$TRAIN_SIZE" <<'PY'
import hashlib, json, pathlib, socket, subprocess, sys
(upstream, tool, stage1_root, ladder_root, stage1_audit, ladder_audit,
 timing_plan, addendum, manifest, corpus_audit, data_sha, no_ttt_config,
 ttt_config, config_audit, start_models, output, steps, seq_len, global_batch,
 train_size) = sys.argv[1:]
sha = lambda path: hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
metadata = {
    "protocol": "in_place_ttt_2x5090_stage2_timing_v1",
    "hostname": socket.gethostname(),
    "upstream_commit": subprocess.check_output(
        ["git", "-C", upstream, "rev-parse", "HEAD"], text=True).strip(),
    "upstream_clean": not subprocess.check_output(
        ["git", "-C", upstream, "status", "--short"], text=True).strip(),
    "tool_commit": subprocess.check_output(
        ["git", "-C", tool, "rev-parse", "HEAD"], text=True).strip(),
    "tool_clean": not subprocess.check_output(
        ["git", "-C", tool, "status", "--short"], text=True).strip(),
    "stage1_root": str(pathlib.Path(stage1_root).resolve()),
    "stage2_ladder_root": str(pathlib.Path(ladder_root).resolve()),
    "steps": int(steps), "sequence_length": int(seq_len),
    "global_batch_size": int(global_batch), "tokens_per_arm": int(train_size),
    "timing_plan_sha256": sha(timing_plan),
    "execution_addendum_sha256": sha(addendum),
    "stage1_audit_sha256": sha(stage1_audit),
    "stage2_ladder_audit_sha256": sha(ladder_audit),
    "corpus_manifest_sha256": sha(manifest),
    "corpus_audit_sha256": sha(corpus_audit), "data_sha256": data_sha,
    "no_ttt_config_sha256": sha(no_ttt_config),
    "ttt_config_sha256": sha(ttt_config),
    "config_audit_sha256": sha(config_audit),
    "start_models_sha256": sha(start_models),
    "runner_sha256": sha(pathlib.Path(tool) / "scripts/run_in_place_ttt_2x5090_stage2_timing.sh"),
    "auditor_sha256": sha(pathlib.Path(tool) / "scripts/audit_in_place_ttt_2x5090_stage2_timing.py"),
}
pathlib.Path(output).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
PY

monitor_pid=""
current_arm=""
write_state() {
  printf '%s\n' "$1" > "$staging/state.txt.partial"
  mv "$staging/state.txt.partial" "$staging/state.txt"
  date --iso-8601=seconds > "$staging/state_time.txt.partial"
  mv "$staging/state_time.txt.partial" "$staging/state_time.txt"
}
stop_monitor() {
  if [[ -n "$monitor_pid" ]]; then
    touch "$staging/$current_arm/monitor.stop"
    wait "$monitor_pid" 2>/dev/null || true
    monitor_pid=""
  fi
}
finish() {
  rc=$?; trap - EXIT; stop_monitor
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

field() {
  "$PYTHON_BIN" - "$staging/start_models.json" "$1" <<'PY'
import json, pathlib, sys
print(json.loads(pathlib.Path(sys.argv[1]).read_text())[sys.argv[2]]["root"])
PY
}
preflight() {
  local root=$1
  nvidia-smi --query-gpu=index,name,memory.total,memory.used \
    --format=csv,noheader,nounits > "$root/preflight_gpu.csv"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader,nounits > "$root/preflight_processes.csv"
  cp /proc/meminfo "$root/preflight_host.txt"; : > "$root/preflight_conflicts.txt"
  "$PYTHON_BIN" - "$root" "$EXPECTED_HOSTNAME" <<'PY'
import csv, pathlib, shlex, socket, sys
root, expected = pathlib.Path(sys.argv[1]), sys.argv[2]
if socket.gethostname() != expected: raise SystemExit("wrong host")
rows = [row for row in csv.reader((root / "preflight_gpu.csv").open()) if row]
if len(rows) != 2: raise SystemExit("expected two GPUs")
for expected_index, row in enumerate(rows):
    index, name, total, used = [item.strip() for item in row]
    if (int(index) != expected_index or "RTX 5090" not in name
            or float(total) < 32000 or float(used) > 128):
        raise SystemExit("GPU capacity/idle gate failed")
if (root / "preflight_processes.csv").read_text().strip():
    raise SystemExit("CUDA process list is nonempty")
targets = {"build_in_place_ttt_public_corpus.py", "audit_in_place_ttt_public_corpus.py"}
conflicts = []
for process in pathlib.Path("/proc").glob("[0-9]*"):
    try:
        argv = [item.decode(errors="replace") for item in
                (process / "cmdline").read_bytes().split(b"\0") if item]
    except (FileNotFoundError, PermissionError, ProcessLookupError): continue
    if any(pathlib.Path(argument).name in targets for argument in argv):
        conflicts.append((int(process.name), argv))
(root / "preflight_conflicts.txt").write_text("".join(
    f"{pid} {shlex.join(argv)}\n" for pid, argv in sorted(conflicts)))
if conflicts: raise SystemExit("corpus process is active")
meminfo = {}
for line in (root / "preflight_host.txt").read_text().splitlines():
    if ":" in line:
        key, rest = line.split(":", 1); meminfo[key] = int(rest.split()[0])
if meminfo.get("MemAvailable", 0) < 48 * 1024 * 1024:
    raise SystemExit("host has less than 48 GiB available")
PY
}

run_arm() {
  local arm=$1 root model config status
  current_arm=$arm; root=$staging/$arm; mkdir -p "$root"; model=$(field "$arm")
  if [[ "$arm" == no_ttt ]]; then config=$NO_TTT_CONFIG; else config=$TTT_CONFIG; fi
  write_state "preflight_${arm}"; preflight "$root"
  printf 'timestamp,index,memory_used_mib,memory_total_mib,utilization_gpu\n' > "$root/gpu.csv"
  printf 'timestamp,memory_total_mib,memory_used_mib,memory_available_mib,swap_total_mib,swap_used_mib\n' > "$root/host.csv"
  (
    while [[ ! -e "$root/monitor.stop" ]]; do
      timestamp=$(date +%s.%N)
      nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader,nounits | while IFS=',' read -r index used total util; do
          printf '%s,%s,%s,%s,%s\n' "$timestamp" "${index// /}" "${used// /}" \
            "${total// /}" "${util// /}" >> "$root/gpu.csv"
        done
      awk -v timestamp="$timestamp" '
        /^MemTotal:/ {total=$2/1024}/^MemAvailable:/ {available=$2/1024}
        /^SwapTotal:/ {swap_total=$2/1024}/^SwapFree:/ {swap_free=$2/1024}
        END {printf "%s,%.3f,%.3f,%.3f,%.3f,%.3f\n", timestamp, total,
                    total-available, available, swap_total, swap_total-swap_free}
      ' /proc/meminfo >> "$root/host.csv"; sleep 1
    done
  ) &
  monitor_pid=$!; write_state "running_${arm}"
  set +e
  (
    cd "$UPSTREAM_REPO"; export PATH="$(dirname "$PYTHON_BIN"):$PATH"
    export PYTHONPATH="$UPSTREAM_REPO${PYTHONPATH:+:$PYTHONPATH}"
    export CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export TOKENIZERS_PARALLELISM=false
    /usr/bin/time -v bash train.sh tasks/train_torch.py "$config" \
      --model.model_path "$model" --data.train_path "$DATA" \
      --data.train_size "$TRAIN_SIZE" --train.output_dir "$root/training_output" \
      --train.max_steps "$STEPS" --train.save_steps 0 --train.save_epochs 0 \
      --train.save_hf_weights false --train.use_wandb false
  ) > "$root/train.log" 2>&1
  status=$?; set -e; printf '%s\n' "$status" > "$root/exit_code.txt"
  if [[ -f "$UPSTREAM_REPO/log.txt" ]]; then mv "$UPSTREAM_REPO/log.txt" "$root/upstream_tee.log"; fi
  stop_monitor
  if [[ "$status" -ne 0 ]]; then write_state "failed_${arm}"; return "$status"; fi
}

run_arm no_ttt
run_arm ttt
write_state auditing
"$PYTHON_BIN" "$TOOL_REPO/scripts/audit_in_place_ttt_2x5090_stage2_timing.py" \
  --root "$staging" --stage1-root "$STAGE1_ROOT" \
  --stage2-ladder-root "$STAGE2_LADDER_ROOT" --timing-plan "$TIMING_PLAN" \
  --execution-addendum "$EXECUTION_ADDENDUM" \
  --corpus-manifest "$CORPUS_MANIFEST" --corpus-audit "$CORPUS_AUDIT" \
  --data "$DATA" --no-ttt-config "$NO_TTT_CONFIG" --ttt-config "$TTT_CONFIG" \
  --runner "$TOOL_REPO/scripts/run_in_place_ttt_2x5090_stage2_timing.sh" \
  --output "$staging/independent_audit.json" > "$staging/independent_audit.stdout"
write_state completed_stage2_timing
