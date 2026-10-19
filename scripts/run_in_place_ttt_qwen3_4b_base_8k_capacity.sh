#!/usr/bin/env bash
set -Eeuo pipefail

UPSTREAM_REPO=${UPSTREAM_REPO:?set the clean official checkout}
TOOL_REPO=${TOOL_REPO:?set the clean CBF-TTT checkout}
PYTHON_BIN=${PYTHON_BIN:?set the official training Python}
MODEL=${MODEL:?set the Qwen3-4B-Base directory}
DATA=${DATA:?set the audited preliminary Stage-1 JSONL}
DATA_MANIFEST=${DATA_MANIFEST:?set the preliminary corpus manifest}
DATA_AUDIT=${DATA_AUDIT:?set the preliminary corpus independent audit}
PLAN=${PLAN:?set the frozen no-headroom capacity addendum}
OUTPUT_ROOT=${OUTPUT_ROOT:?set a new output root}
EXPECTED_HOSTNAME=${EXPECTED_HOSTNAME:-user-Super-Server}
UPSTREAM_COMMIT=be2324829b0e91c8fd10a74d4b43714fde6676e1
CONFIG=$TOOL_REPO/configs/baseline_reproduction/qwen3_4b_2x5090_100m_stage1_ttt.yaml
STEPS=${CAPACITY_STEPS:-10}
SEQ_LEN=${CAPACITY_SEQ_LEN:-8192}
GLOBAL_BATCH_SIZE=2
TRAIN_SIZE=$((STEPS * SEQ_LEN * GLOBAL_BATCH_SIZE))

if [[ "$STEPS" -ne 10 || ("$SEQ_LEN" -ne 4096 && "$SEQ_LEN" -ne 8192) ]]; then
  echo "capacity diagnostic permits 10 steps at 4K or 8K only" >&2; exit 2
fi

if [[ "$(hostname)" != "$EXPECTED_HOSTNAME" ]]; then
  echo "capacity diagnostic is restricted to hku-gpu2" >&2; exit 2
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "refusing to overwrite $OUTPUT_ROOT" >&2; exit 3
fi
test "$(git -C "$UPSTREAM_REPO" rev-parse HEAD)" = "$UPSTREAM_COMMIT"
test -z "$(git -C "$UPSTREAM_REPO" status --short)"
test -z "$(git -C "$TOOL_REPO" status --short)"
for path in "$PYTHON_BIN" "$MODEL/config.json" "$DATA" "$DATA_MANIFEST" \
  "$DATA_AUDIT" "$PLAN" "$CONFIG"; do test -e "$path"; done
if [[ "$(df -PB1 "$(dirname "$OUTPUT_ROOT")" | awk 'NR==2 {print $4}')" -lt 80000000000 ]]; then
  echo "capacity diagnostic requires at least 80 GB free disk" >&2; exit 4
fi

mkdir -p "$OUTPUT_ROOT/checkpoint"
write_status() {
  printf '%s\n' "$1" > "$OUTPUT_ROOT/status.txt.partial"
  mv "$OUTPUT_ROOT/status.txt.partial" "$OUTPUT_ROOT/status.txt"
}
write_status PREFLIGHT

nvidia-smi --query-gpu=index,name,memory.total,memory.used \
  --format=csv,noheader,nounits > "$OUTPUT_ROOT/preflight_gpu.csv"
nvidia-smi --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader,nounits > "$OUTPUT_ROOT/preflight_processes.csv"
cp /proc/meminfo "$OUTPUT_ROOT/preflight_host.txt"
"$PYTHON_BIN" - "$OUTPUT_ROOT" <<'PY'
import csv, pathlib, sys
root = pathlib.Path(sys.argv[1])
rows = [row for row in csv.reader((root / "preflight_gpu.csv").open()) if row]
if len(rows) != 2:
    raise SystemExit("expected exactly two GPUs")
for expected, row in enumerate(rows):
    index, name, total, used = [item.strip() for item in row]
    if (int(index) != expected or "RTX 5090" not in name
            or float(total) < 32000 or float(used) > 128):
        raise SystemExit(f"GPU is not exclusively idle: {row}")
if (root / "preflight_processes.csv").read_text().strip():
    raise SystemExit("another CUDA compute process is active")
mem = {}
for line in (root / "preflight_host.txt").read_text().splitlines():
    if ":" in line:
        key, rest = line.split(":", 1)
        mem[key] = int(rest.split()[0])
if mem.get("MemAvailable", 0) < 48 * 1024 * 1024:
    raise SystemExit("less than 48 GiB host memory is available")
PY

"$PYTHON_BIN" - "$UPSTREAM_REPO" "$TOOL_REPO" "$MODEL" "$DATA" \
  "$DATA_MANIFEST" "$DATA_AUDIT" "$PLAN" "$CONFIG" \
  "$OUTPUT_ROOT/run_metadata.json" "$STEPS" "$SEQ_LEN" \
  "$GLOBAL_BATCH_SIZE" "$TRAIN_SIZE" <<'PY'
import hashlib, json, pathlib, socket, subprocess, sys
(upstream, tool, model, data, manifest, audit, plan, config, output,
 steps, seq_len, batch, train_size) = sys.argv[1:]
def sha(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()
payload = {
    "protocol": "qwen3_4b_base_2x5090_no_headroom_capacity_v1",
    "hostname": socket.gethostname(),
    "upstream_commit": subprocess.check_output(
        ["git", "-C", upstream, "rev-parse", "HEAD"], text=True).strip(),
    "tool_commit": subprocess.check_output(
        ["git", "-C", tool, "rev-parse", "HEAD"], text=True).strip(),
    "model_config_sha256": sha(pathlib.Path(model) / "config.json"),
    "data_sha256": sha(data),
    "data_manifest_sha256": sha(manifest),
    "data_audit_sha256": sha(audit),
    "plan_sha256": sha(plan),
    "config_sha256": sha(config),
    "steps": int(steps),
    "sequence_length": int(seq_len),
    "global_batch_size": int(batch),
    "train_size": int(train_size),
    "headroom_is_stopping_rule": False,
}
pathlib.Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
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
      /^MemTotal:/ {total=$2/1024}/^MemAvailable:/ {available=$2/1024}
      /^SwapTotal:/ {swap_total=$2/1024}/^SwapFree:/ {swap_free=$2/1024}
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

write_status RUNNING
set +e
(
  cd "$UPSTREAM_REPO"
  export PATH="$(dirname "$PYTHON_BIN"):$PATH"
  export PYTHONPATH="$UPSTREAM_REPO${PYTHONPATH:+:$PYTHONPATH}"
  export CUDA_VISIBLE_DEVICES=0,1
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export TOKENIZERS_PARALLELISM=false
  /usr/bin/time -v bash train.sh tasks/train_torch.py "$CONFIG" \
    --model.model_path "$MODEL" \
    --data.train_path "$DATA" --data.max_seq_len "$SEQ_LEN" \
    --data.train_size "$TRAIN_SIZE" \
    --train.output_dir "$OUTPUT_ROOT/checkpoint" \
    --train.global_batch_size "$GLOBAL_BATCH_SIZE" \
    --train.micro_batch_size 1 --train.max_steps "$STEPS" \
    --train.save_steps "$STEPS" --train.save_epochs 0 \
    --train.save_hf_weights false --train.use_wandb false
) > "$OUTPUT_ROOT/train.log" 2>&1
status=$?
set -e
printf '%s\n' "$status" > "$OUTPUT_ROOT/exit_code.txt"
if [[ -f "$UPSTREAM_REPO/log.txt" ]]; then
  mv "$UPSTREAM_REPO/log.txt" "$OUTPUT_ROOT/upstream_tee.log"
fi
cleanup
trap - EXIT

set +e
"$PYTHON_BIN" - "$OUTPUT_ROOT" "$STEPS" <<'PY'
import csv, json, math, pathlib, re, sys
root, expected_steps = pathlib.Path(sys.argv[1]), int(sys.argv[2])
text = (root / "train.log").read_text(errors="replace")
exit_code = int((root / "exit_code.txt").read_text())
losses = [float(x) for x in re.findall(r"['\"]loss['\"]:\s*([0-9.eE+-]+)", text)]
gradients = [float(x) for x in re.findall(
    r"['\"]gradient_norm['\"]:\s*([0-9.eE+-]+)", text)]
step_numbers = [int(x) for x in re.findall(
    r"['\"]global_step['\"]:\s*(\d+)", text)]
checkpoint_bytes = sum(
    p.stat().st_size for p in (root / "checkpoint").rglob("*") if p.is_file())
gpu_rows = list(csv.DictReader((root / "gpu.csv").open()))
host_rows = list(csv.DictReader((root / "host.csv").open()))
peaks = {}
for row in gpu_rows:
    index = row["index"]
    peaks[index] = max(peaks.get(index, 0.0), float(row["memory_used_mib"]))
totals = {row["index"]: float(row["memory_total_mib"]) for row in gpu_rows}
minimum_free = {
    index: (totals[index] - used) / totals[index]
    for index, used in peaks.items()
}
oom = "out of memory" in text.lower() or "cuda error" in text.lower()
nonfinite = any(not math.isfinite(x) for x in losses + gradients)
completed_steps = max(step_numbers, default=len(losses))
passed = (
    exit_code == 0 and completed_steps >= expected_steps and losses
    and gradients and not nonfinite and not oom and checkpoint_bytes > 0
)
summary = {
    "protocol": "qwen3_4b_base_2x5090_no_headroom_capacity_result_v1",
    "passed": bool(passed),
    "exit_code": exit_code,
    "expected_steps": expected_steps,
    "completed_steps": completed_steps,
    "last_loss": losses[-1] if losses else None,
    "last_gradient_norm": gradients[-1] if gradients else None,
    "oom_signature": oom,
    "nonfinite": nonfinite,
    "checkpoint_bytes": checkpoint_bytes,
    "peak_gpu_memory_mib": peaks,
    "minimum_gpu_free_fraction": minimum_free,
    "minimum_host_available_mib": min(
        (float(row["memory_available_mib"]) for row in host_rows), default=None),
    "peak_swap_used_mib": max(
        (float(row["swap_used_mib"]) for row in host_rows), default=None),
    "headroom_is_stopping_rule": False,
}
(root / "summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n")
if not passed:
    raise SystemExit("two-GPU no-headroom capacity diagnostic failed")
PY
summary_status=$?
set -e
if [[ "$summary_status" -eq 0 ]]; then
  write_status COMPLETE
else
  write_status FAILED
  exit "$summary_status"
fi
