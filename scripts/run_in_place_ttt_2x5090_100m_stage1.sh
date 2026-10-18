#!/usr/bin/env bash
set -Eeuo pipefail

UPSTREAM_REPO=${UPSTREAM_REPO:?set the clean official checkout}
TOOL_REPO=${TOOL_REPO:?set the clean CBF-TTT checkout}
TOOL_COMMIT=${TOOL_COMMIT:?set the exact tool commit}
PYTHON_BIN=${PYTHON_BIN:?set the official training Python}
MODEL=${MODEL:?set the Qwen3-4B model directory}
CORPUS_ROOT=${CORPUS_ROOT:?set the independently audited full corpus}
TIMING_ROOT=${TIMING_ROOT:?set the completed paired Stage-1 timing root}
RELEASE=${RELEASE:?set the explicit formal Stage-1 release JSON}
OUTPUT_ROOT=${OUTPUT_ROOT:?set a new paired Stage-1 output root}
FORMAL_PLAN=${FORMAL_PLAN:?set the frozen formal Stage-1 plan}
EXECUTION_ADDENDUM=${EXECUTION_ADDENDUM:?set the frozen 2x5090 addendum}
EXPECTED_HOSTNAME=${EXPECTED_HOSTNAME:-user-Super-Server}
UPSTREAM_COMMIT=be2324829b0e91c8fd10a74d4b43714fde6676e1
STEPS=27
SEQ_LEN=32768
GLOBAL_BATCH_SIZE=64
TRAIN_SIZE=56623104
DATA=$CORPUS_ROOT/stage1_32k.jsonl
CORPUS_MANIFEST=$CORPUS_ROOT/manifest.json
CORPUS_AUDIT=$CORPUS_ROOT/independent_audit.json
TIMING_AUDIT=$TIMING_ROOT/independent_audit.json
NO_TTT_CONFIG=$TOOL_REPO/configs/baseline_reproduction/qwen3_4b_2x5090_100m_stage1_no_ttt.yaml
TTT_CONFIG=$TOOL_REPO/configs/baseline_reproduction/qwen3_4b_2x5090_100m_stage1_ttt.yaml

if [[ "$(hostname)" != "$EXPECTED_HOSTNAME" ]]; then
  echo "formal Stage-1 training is restricted to hku-gpu2" >&2; exit 2
fi
if [[ -e "$OUTPUT_ROOT" || -e "${OUTPUT_ROOT}.partial" ]]; then
  echo "refusing to overwrite formal Stage-1 output" >&2; exit 3
fi
test "$(git -C "$UPSTREAM_REPO" rev-parse HEAD)" = "$UPSTREAM_COMMIT"
test -z "$(git -C "$UPSTREAM_REPO" status --short)"
test "$(git -C "$TOOL_REPO" rev-parse HEAD)" = "$TOOL_COMMIT"
test -z "$(git -C "$TOOL_REPO" status --short)"
test -x "$PYTHON_BIN"; test -s "$MODEL/config.json"
test -s "$DATA"; test -s "$CORPUS_MANIFEST"; test -s "$CORPUS_AUDIT"
test -s "$TIMING_AUDIT"; test -s "$RELEASE"
test -s "$FORMAL_PLAN"; test -s "$EXECUTION_ADDENDUM"
test -s "$NO_TTT_CONFIG"; test -s "$TTT_CONFIG"

disk_pair=$(df -PB1 "$(dirname "$OUTPUT_ROOT")" | awk 'NR==2 {print $4}')
if [[ "$disk_pair" -lt 160000000000 ]]; then
  echo "formal Stage-1 pair requires at least 160 GB free disk" >&2; exit 4
fi

DATA_SHA=$("$PYTHON_BIN" - "$DATA" "$CORPUS_MANIFEST" "$CORPUS_AUDIT" \
  "$TIMING_AUDIT" "$RELEASE" <<'PY'
import hashlib, json, math, pathlib, sys
data, manifest_path, corpus_audit_path, timing_path, release_path = map(
    pathlib.Path, sys.argv[1:]
)
def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()
manifest = json.loads(manifest_path.read_text())
corpus_audit = json.loads(corpus_audit_path.read_text())
timing = json.loads(timing_path.read_text())
release = json.loads(release_path.read_text())
stage = next((item for item in manifest.get("stages", [])
              if item.get("name") == "stage1_32k"), {})
artifact = stage.get("artifacts", {}).get(data.name, {})
data_sha = sha(data)
if (manifest.get("protocol") != "in_place_ttt_public_corpus_v1"
        or manifest.get("profile") != "full"
        or corpus_audit.get("passed") is not True
        or corpus_audit.get("profile") != "full"
        or corpus_audit.get("bindings", {}).get("manifest_sha256") != sha(manifest_path)
        or artifact.get("sha256") != data_sha
        or artifact.get("size_bytes") != data.stat().st_size):
    raise SystemExit("full corpus did not release formal Stage 1")
projection = timing.get("projected_sequential_stage1_seconds")
if (timing.get("protocol") != "in_place_ttt_2x5090_stage1_timing_audit_v1"
        or timing.get("passed") is not True
        or not isinstance(projection, (int, float))
        or not math.isfinite(projection) or projection <= 0):
    raise SystemExit("paired timing audit did not release formal Stage 1")
if (release.get("protocol") != "in_place_ttt_2x5090_100m_stage1_release_v1"
        or release.get("accepted") is not True
        or release.get("timing_audit_sha256") != sha(timing_path)):
    raise SystemExit("explicit timing-bound release is absent or invalid")
print(data_sha)
PY
)

staging=${OUTPUT_ROOT}.partial
mkdir -p "$staging"
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
  rc=$?
  trap - EXIT
  stop_monitor
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

"$PYTHON_BIN" "$TOOL_REPO/scripts/audit_in_place_ttt_2x5090_configs.py" \
  --output "$staging/config_audit.json" > "$staging/config_audit.stdout"

"$PYTHON_BIN" - "$UPSTREAM_REPO" "$TOOL_REPO" "$MODEL" "$DATA_SHA" \
  "$CORPUS_MANIFEST" "$CORPUS_AUDIT" "$TIMING_AUDIT" "$RELEASE" \
  "$FORMAL_PLAN" "$EXECUTION_ADDENDUM" "$NO_TTT_CONFIG" "$TTT_CONFIG" \
  "$staging/config_audit.json" "$staging/run_metadata.json" \
  "$STEPS" "$SEQ_LEN" "$GLOBAL_BATCH_SIZE" "$TRAIN_SIZE" "$disk_pair" <<'PY'
import hashlib, json, pathlib, socket, subprocess, sys
(upstream, tool, model, data_sha, manifest, corpus_audit, timing_audit, release,
 formal_plan, addendum, no_ttt_config, ttt_config, config_audit, output, steps,
 seq_len, global_batch, train_size, disk_pair) = sys.argv[1:]
sha = lambda path: hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
metadata = {
    "protocol": "in_place_ttt_2x5090_100m_stage1_v1",
    "hostname": socket.gethostname(),
    "upstream_commit": subprocess.check_output(
        ["git", "-C", upstream, "rev-parse", "HEAD"], text=True).strip(),
    "upstream_clean": not subprocess.check_output(
        ["git", "-C", upstream, "status", "--short"], text=True).strip(),
    "tool_commit": subprocess.check_output(
        ["git", "-C", tool, "rev-parse", "HEAD"], text=True).strip(),
    "tool_clean": not subprocess.check_output(
        ["git", "-C", tool, "status", "--short"], text=True).strip(),
    "steps_per_arm": int(steps), "sequence_length": int(seq_len),
    "global_batch_size": int(global_batch), "tokens_per_arm": int(train_size),
    "disk_free_bytes_before_pair": int(disk_pair),
    "formal_plan_sha256": sha(formal_plan),
    "execution_addendum_sha256": sha(addendum),
    "timing_audit_sha256": sha(timing_audit),
    "release_sha256": sha(release),
    "corpus_manifest_sha256": sha(manifest),
    "corpus_audit_sha256": sha(corpus_audit),
    "data_sha256": data_sha,
    "model_config_sha256": sha(pathlib.Path(model) / "config.json"),
    "no_ttt_config_sha256": sha(no_ttt_config),
    "ttt_config_sha256": sha(ttt_config),
    "config_audit_sha256": sha(config_audit),
    "runner_sha256": sha(pathlib.Path(tool) / "scripts/run_in_place_ttt_2x5090_100m_stage1.sh"),
    "auditor_sha256": sha(pathlib.Path(tool) / "scripts/audit_in_place_ttt_2x5090_100m_stage1.py"),
}
pathlib.Path(output).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
PY

preflight() {
  local root=$1
  nvidia-smi --query-gpu=index,name,memory.total,memory.used \
    --format=csv,noheader,nounits > "$root/preflight_gpu.csv"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader,nounits > "$root/preflight_processes.csv"
  cp /proc/meminfo "$root/preflight_host.txt"
  : > "$root/preflight_conflicts.txt"
  "$PYTHON_BIN" - "$root" "$EXPECTED_HOSTNAME" <<'PY'
import csv, pathlib, shlex, socket, sys
root, expected = pathlib.Path(sys.argv[1]), sys.argv[2]
if socket.gethostname() != expected:
    raise SystemExit("wrong host")
rows = [row for row in csv.reader((root / "preflight_gpu.csv").open()) if row]
if len(rows) != 2:
    raise SystemExit("expected exactly two GPUs")
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
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    if any(pathlib.Path(argument).name in targets for argument in argv):
        conflicts.append((int(process.name), argv))
(root / "preflight_conflicts.txt").write_text("".join(
    f"{pid} {shlex.join(argv)}\n" for pid, argv in sorted(conflicts)))
if conflicts:
    raise SystemExit("public-corpus builder/auditor is active")
meminfo = {}
for line in (root / "preflight_host.txt").read_text().splitlines():
    if ":" in line:
        key, rest = line.split(":", 1); meminfo[key] = int(rest.split()[0])
if meminfo.get("MemAvailable", 0) < 48 * 1024 * 1024:
    raise SystemExit("host has less than 48 GiB available")
PY
}

run_arm() {
  local arm=$1 config root status disk_free
  current_arm=$arm
  root=$staging/$arm
  mkdir -p "$root"
  if [[ "$arm" == no_ttt ]]; then config=$NO_TTT_CONFIG; else config=$TTT_CONFIG; fi
  disk_free=$(df -PB1 "$staging" | awk 'NR==2 {print $4}')
  printf '%s\n' "$disk_free" > "$root/disk_before.txt"
  if [[ "$disk_free" -lt 80000000000 ]]; then
    echo "$arm requires at least 80 GB free disk" >&2; return 5
  fi
  write_state "preflight_${arm}"
  preflight "$root"
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
      ' /proc/meminfo >> "$root/host.csv"
      sleep 1
    done
  ) &
  monitor_pid=$!
  write_state "running_${arm}"
  set +e
  (
    cd "$UPSTREAM_REPO"
    export PATH="$(dirname "$PYTHON_BIN"):$PATH"
    export PYTHONPATH="$UPSTREAM_REPO${PYTHONPATH:+:$PYTHONPATH}"
    export CUDA_VISIBLE_DEVICES=0,1
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export TOKENIZERS_PARALLELISM=false
    /usr/bin/time -v bash train.sh tasks/train_torch.py "$config" \
      --model.model_path "$MODEL" --data.train_path "$DATA" \
      --train.output_dir "$root/training_output" --train.use_wandb false
  ) > "$root/train.log" 2>&1
  status=$?
  set -e
  printf '%s\n' "$status" > "$root/exit_code.txt"
  if [[ -f "$UPSTREAM_REPO/log.txt" ]]; then
    mv "$UPSTREAM_REPO/log.txt" "$root/upstream_tee.log"
  fi
  stop_monitor
  if [[ "$status" -ne 0 ]]; then
    write_state "failed_${arm}"
    return "$status"
  fi
}

run_arm no_ttt
run_arm ttt
write_state auditing
"$PYTHON_BIN" "$TOOL_REPO/scripts/audit_in_place_ttt_2x5090_100m_stage1.py" \
  --root "$staging" --formal-plan "$FORMAL_PLAN" \
  --execution-addendum "$EXECUTION_ADDENDUM" --timing-audit "$TIMING_AUDIT" \
  --release "$RELEASE" --corpus-manifest "$CORPUS_MANIFEST" \
  --corpus-audit "$CORPUS_AUDIT" --data "$DATA" \
  --model-config "$MODEL/config.json" --no-ttt-config "$NO_TTT_CONFIG" \
  --ttt-config "$TTT_CONFIG" \
  --runner "$TOOL_REPO/scripts/run_in_place_ttt_2x5090_100m_stage1.sh" \
  --output "$staging/independent_audit.json" > "$staging/independent_audit.stdout"
write_state completed_100m_stage1
