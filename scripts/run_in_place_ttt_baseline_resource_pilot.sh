#!/usr/bin/env bash
set -euo pipefail

UPSTREAM_REPO=${UPSTREAM_REPO:?set the clean official checkout}
PYTHON_BIN=${PYTHON_BIN:?set the baseline training Python}
MODEL=${MODEL:?set the Qwen3-4B model directory}
DATA=${DATA:?set the packed pilot JSONL}
OUTPUT_ROOT=${OUTPUT_ROOT:?set a new output root}
ARM=${ARM:?set ARM to ttt or no_ttt}
RESOURCE_PROFILE=${RESOURCE_PROFILE:-native}
SEQ_LEN=${SEQ_LEN:-8192}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-2}
PLAN=${PLAN:?set the frozen baseline plan path}
TOOL_REPO=${TOOL_REPO:?set the CBF-TTT tool checkout}
ADDENDUM=${ADDENDUM:-}
CONFIG_OVERRIDE=${CONFIG_OVERRIDE:-}
EXPECTED_HOSTNAME=${EXPECTED_HOSTNAME:-user-Super-Server}
UPSTREAM_COMMIT=be2324829b0e91c8fd10a74d4b43714fde6676e1

if [[ "$ARM" != "ttt" && "$ARM" != "no_ttt" ]]; then
  echo "ARM must be ttt or no_ttt" >&2; exit 2
fi
if [[ "$RESOURCE_PROFILE" != "native" && "$RESOURCE_PROFILE" != "cpu_offload" \
      && "$RESOURCE_PROFILE" != "fsdp1_cpu_offload" \
      && "$RESOURCE_PROFILE" != "native_expandable_segments" \
      && "$RESOURCE_PROFILE" != "activation_offload" ]]; then
  echo "invalid RESOURCE_PROFILE" >&2; exit 2
fi
if [[ "$SEQ_LEN" -lt 8192 || $((SEQ_LEN % 4096)) -ne 0 ]]; then
  echo "SEQ_LEN must be at least 8192 and divisible by 4096" >&2; exit 2
fi
if [[ "$GLOBAL_BATCH_SIZE" -ne 2 ]]; then
  echo "the registered two-GPU pilot requires GLOBAL_BATCH_SIZE=2" >&2; exit 2
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "refusing to overwrite $OUTPUT_ROOT" >&2; exit 3
fi
test "$(git -C "$UPSTREAM_REPO" rev-parse HEAD)" = "$UPSTREAM_COMMIT"
test -z "$(git -C "$UPSTREAM_REPO" status --short)"
test -s "$MODEL/config.json"; test -s "$DATA"; test -s "$PLAN"

config_suffix=""
if [[ "$RESOURCE_PROFILE" == "cpu_offload" ]]; then
  config_suffix=_cpu_offload
  test -n "$ADDENDUM"; test -s "$ADDENDUM"
elif [[ "$RESOURCE_PROFILE" == "fsdp1_cpu_offload" ]]; then
  config_suffix=_fsdp1_cpu_offload
  test -n "$ADDENDUM"; test -s "$ADDENDUM"
elif [[ "$RESOURCE_PROFILE" == "native_expandable_segments" ]]; then
  test -n "$ADDENDUM"; test -s "$ADDENDUM"
elif [[ "$RESOURCE_PROFILE" == "activation_offload" ]]; then
  test -n "$ADDENDUM"; test -s "$ADDENDUM"
  test -n "$CONFIG_OVERRIDE"
fi
if [[ -n "$CONFIG_OVERRIDE" ]]; then
  CONFIG=$CONFIG_OVERRIDE
else
  CONFIG=$TOOL_REPO/configs/baseline_reproduction/qwen3_4b_resource_${ARM}${config_suffix}.yaml
fi
test -s "$CONFIG"
mkdir -p "$OUTPUT_ROOT/checkpoint"

# The activation-offload path is registered only for the two RTX 5090 devices
# on hku-gpu2.  Refuse to start while the public-corpus builder/auditor is using
# host RAM, or when another CUDA process could contaminate the resource trace.
if [[ "$RESOURCE_PROFILE" == "activation_offload" ]]; then
  test "$(hostname)" = "$EXPECTED_HOSTNAME"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used \
    --format=csv,noheader,nounits > "$OUTPUT_ROOT/preflight_gpu.csv"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader,nounits > "$OUTPUT_ROOT/preflight_processes.csv"
  cp /proc/meminfo "$OUTPUT_ROOT/preflight_host.txt"
  : > "$OUTPUT_ROOT/preflight_conflicts.txt"
  "$PYTHON_BIN" - "$OUTPUT_ROOT" "$EXPECTED_HOSTNAME" <<'PY'
import csv
import pathlib
import shlex
import socket
import sys

root, expected_hostname = pathlib.Path(sys.argv[1]), sys.argv[2]
if socket.gethostname() != expected_hostname:
    raise SystemExit("activation-offload pilot is restricted to hku-gpu2")
with (root / "preflight_gpu.csv").open(newline="") as handle:
    rows = [row for row in csv.reader(handle) if row]
if len(rows) != 2:
    raise SystemExit(f"expected exactly two GPUs, observed {len(rows)}")
for expected_index, row in enumerate(rows):
    index, name, total, used = [value.strip() for value in row]
    if int(index) != expected_index or "RTX 5090" not in name:
        raise SystemExit(f"unexpected hku-gpu2 GPU binding: {row}")
    if float(total) < 32_000 or float(used) > 128:
        raise SystemExit(f"GPU {index} is not exclusively idle: {row}")
if (root / "preflight_processes.csv").read_text().strip():
    raise SystemExit("another CUDA compute process is active")
targets = {
    "build_in_place_ttt_public_corpus.py",
    "audit_in_place_ttt_public_corpus.py",
}
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
    f"{pid} {shlex.join(argv)}\n" for pid, argv in sorted(conflicts)
))
if (root / "preflight_conflicts.txt").read_text().strip():
    raise SystemExit("public-corpus builder/auditor is still active")
meminfo = {}
for line in (root / "preflight_host.txt").read_text().splitlines():
    if ":" in line:
        key, rest = line.split(":", 1)
        meminfo[key] = int(rest.split()[0])
if meminfo.get("MemAvailable", 0) < 48 * 1024 * 1024:
    raise SystemExit("hku-gpu2 has less than 48 GiB host memory available")
PY
fi

"$PYTHON_BIN" - "$UPSTREAM_REPO" "$MODEL" "$DATA" "$CONFIG" "$PLAN" \
  "$OUTPUT_ROOT/run_metadata.json" "$ARM" "$SEQ_LEN" "$GLOBAL_BATCH_SIZE" \
  "$RESOURCE_PROFILE" "$ADDENDUM" "$TOOL_REPO" <<'PY'
import hashlib, importlib, json, pathlib, socket, subprocess, sys
upstream, model, data, config, plan, output, arm, seq_len, batch, profile, addendum, tools = sys.argv[1:]
sha = lambda path: hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
root = pathlib.Path(output).parent
offloading_path = None
if profile == "activation_offload":
    module = importlib.import_module("veomni.distributed.offloading")
    offloading_path = pathlib.Path(module.__file__).resolve()
metadata = {
    "protocol": "in_place_ttt_baseline_resource_v2",
    "upstream_commit": subprocess.check_output(
        ["git", "-C", upstream, "rev-parse", "HEAD"], text=True).strip(),
    "upstream_clean": not subprocess.check_output(
        ["git", "-C", upstream, "status", "--short"], text=True).strip(),
    "hostname": socket.gethostname(),
    "model": model, "data": data, "config": config, "arm": arm,
    "resource_profile": profile,
    "activation_offload": profile == "activation_offload",
    "pytorch_cuda_alloc_conf": (
        "expandable_segments:True"
        if profile in {"native_expandable_segments", "activation_offload"} else None
    ),
    "sequence_length": int(seq_len), "global_batch_size": int(batch),
    "plan_sha256": sha(plan), "config_sha256": sha(config),
    "data_sha256": sha(data), "model_config_sha256": sha(pathlib.Path(model) / "config.json"),
    "addendum_sha256": sha(addendum) if addendum else None,
    "activation_offloading_path": str(offloading_path) if offloading_path else None,
    "activation_offloading_sha256": sha(offloading_path) if offloading_path else None,
    "preflight_gpu_sha256": sha(root / "preflight_gpu.csv") if profile == "activation_offload" else None,
    "preflight_processes_sha256": sha(root / "preflight_processes.csv") if profile == "activation_offload" else None,
    "preflight_host_sha256": sha(root / "preflight_host.txt") if profile == "activation_offload" else None,
    "preflight_conflicts_sha256": sha(root / "preflight_conflicts.txt") if profile == "activation_offload" else None,
    "runner_sha256": sha(pathlib.Path(tools) / "scripts/run_in_place_ttt_baseline_resource_pilot.sh"),
    "summarizer_sha256": sha(pathlib.Path(tools) / "scripts/summarize_in_place_ttt_baseline_resource.py"),
}
pathlib.Path(output).write_text(json.dumps(metadata, indent=2) + "\n")
PY

printf 'timestamp,index,memory_used_mib,memory_total_mib,utilization_gpu\n' > "$OUTPUT_ROOT/gpu.csv"
printf 'timestamp,memory_total_mib,memory_used_mib,memory_available_mib,swap_total_mib,swap_used_mib\n' > "$OUTPUT_ROOT/host.csv"
(
  while [[ ! -e "$OUTPUT_ROOT/monitor.stop" ]]; do
    timestamp=$(date +%s.%N)
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits | while IFS=',' read -r index used total util; do
        printf '%s,%s,%s,%s,%s\n' "$timestamp" "${index// /}" "${used// /}" \
          "${total// /}" "${util// /}" >> "$OUTPUT_ROOT/gpu.csv"
      done
    awk -v timestamp="$timestamp" '
      /^MemTotal:/ {total=$2/1024}
      /^MemAvailable:/ {available=$2/1024}
      /^SwapTotal:/ {swap_total=$2/1024}
      /^SwapFree:/ {swap_free=$2/1024}
      END {printf "%s,%.3f,%.3f,%.3f,%.3f,%.3f\n", timestamp, total,
                  total-available, available, swap_total, swap_total-swap_free}
    ' /proc/meminfo >> "$OUTPUT_ROOT/host.csv"
    sleep 0.5
  done
) &
monitor_pid=$!
cleanup() {
  touch "$OUTPUT_ROOT/monitor.stop"
  wait "$monitor_pid" 2>/dev/null || true
}
trap cleanup EXIT

set +e
(
  cd "$UPSTREAM_REPO"
  export PATH="$(dirname "$PYTHON_BIN"):$PATH"
  # The released train.sh invokes tasks/train_torch.py as a file, so its
  # sibling top-level hf_models package is visible only when the checkout is
  # explicitly present on PYTHONPATH.
  export PYTHONPATH="$UPSTREAM_REPO${PYTHONPATH:+:$PYTHONPATH}"
  export CUDA_VISIBLE_DEVICES=0,1
  if [[ "$RESOURCE_PROFILE" == "native_expandable_segments" \
        || "$RESOURCE_PROFILE" == "activation_offload" ]]; then
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  else
    unset PYTORCH_CUDA_ALLOC_CONF || true
  fi
  /usr/bin/time -v bash train.sh tasks/train_torch.py "$CONFIG" \
    --model.model_path "$MODEL" \
    --data.train_path "$DATA" --data.max_seq_len "$SEQ_LEN" \
    --data.train_size "$((SEQ_LEN * GLOBAL_BATCH_SIZE))" \
    --train.output_dir "$OUTPUT_ROOT/checkpoint" \
    --train.global_batch_size "$GLOBAL_BATCH_SIZE" --train.max_steps 1 \
    --train.save_steps 1 --train.save_epochs 0 --train.save_hf_weights false \
    --train.use_wandb false
) > "$OUTPUT_ROOT/train.log" 2>&1
status=$?
set -e
printf '%s\n' "$status" > "$OUTPUT_ROOT/exit_code.txt"
if [[ -f "$UPSTREAM_REPO/log.txt" ]]; then
  mv "$UPSTREAM_REPO/log.txt" "$OUTPUT_ROOT/upstream_tee.log"
fi
cleanup
trap - EXIT

"$PYTHON_BIN" "$TOOL_REPO/scripts/summarize_in_place_ttt_baseline_resource.py" \
  --root "$OUTPUT_ROOT" --output "$OUTPUT_ROOT/summary.json"
audit_status=0
if [[ "$RESOURCE_PROFILE" == "activation_offload" ]]; then
  offloading_path=$("$PYTHON_BIN" - "$OUTPUT_ROOT/run_metadata.json" <<'PY'
import json, pathlib, sys
print(json.loads(pathlib.Path(sys.argv[1]).read_text())["activation_offloading_path"])
PY
)
  set +e
  "$PYTHON_BIN" "$TOOL_REPO/scripts/audit_in_place_ttt_2x5090_resource.py" \
    --root "$OUTPUT_ROOT" --plan "$PLAN" --addendum "$ADDENDUM" \
    --config "$CONFIG" --data "$DATA" --model-config "$MODEL/config.json" \
    --runner "$TOOL_REPO/scripts/run_in_place_ttt_baseline_resource_pilot.sh" \
    --summarizer "$TOOL_REPO/scripts/summarize_in_place_ttt_baseline_resource.py" \
    --offloading "$offloading_path" \
    --output "$OUTPUT_ROOT/independent_audit.json"
  audit_status=$?
  set -e
fi
if [[ "$status" -ne 0 ]]; then
  exit "$status"
fi
exit "$audit_status"

