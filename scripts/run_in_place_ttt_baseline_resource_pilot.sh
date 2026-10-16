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
UPSTREAM_COMMIT=be2324829b0e91c8fd10a74d4b43714fde6676e1

if [[ "$ARM" != "ttt" && "$ARM" != "no_ttt" ]]; then
  echo "ARM must be ttt or no_ttt" >&2; exit 2
fi
if [[ "$RESOURCE_PROFILE" != "native" && "$RESOURCE_PROFILE" != "cpu_offload" ]]; then
  echo "RESOURCE_PROFILE must be native or cpu_offload" >&2; exit 2
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
fi
CONFIG=$TOOL_REPO/configs/baseline_reproduction/qwen3_4b_resource_${ARM}${config_suffix}.yaml
test -s "$CONFIG"
mkdir -p "$OUTPUT_ROOT/checkpoint"

"$PYTHON_BIN" - "$UPSTREAM_REPO" "$MODEL" "$DATA" "$CONFIG" "$PLAN" \
  "$OUTPUT_ROOT/run_metadata.json" "$ARM" "$SEQ_LEN" "$GLOBAL_BATCH_SIZE" \
  "$RESOURCE_PROFILE" "$ADDENDUM" "$TOOL_REPO" <<'PY'
import hashlib, json, pathlib, subprocess, sys
upstream, model, data, config, plan, output, arm, seq_len, batch, profile, addendum, tools = sys.argv[1:]
sha = lambda path: hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
metadata = {
    "protocol": "in_place_ttt_baseline_resource_v2",
    "upstream_commit": subprocess.check_output(
        ["git", "-C", upstream, "rev-parse", "HEAD"], text=True).strip(),
    "upstream_clean": not subprocess.check_output(
        ["git", "-C", upstream, "status", "--short"], text=True).strip(),
    "model": model, "data": data, "arm": arm,
    "resource_profile": profile,
    "sequence_length": int(seq_len), "global_batch_size": int(batch),
    "plan_sha256": sha(plan), "config_sha256": sha(config),
    "data_sha256": sha(data), "model_config_sha256": sha(pathlib.Path(model) / "config.json"),
    "addendum_sha256": sha(addendum) if addendum else None,
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
  /usr/bin/time -v bash train.sh tasks/train_torch.py "$CONFIG" \
    --model.model_path "$MODEL" \
    --data.train_path "$DATA" --data.max_seq_len "$SEQ_LEN" \
    --data.train_size "$((SEQ_LEN * GLOBAL_BATCH_SIZE))" \
    --train.output_dir "$OUTPUT_ROOT/checkpoint" \
    --train.global_batch_size "$GLOBAL_BATCH_SIZE" --train.max_steps 1 \
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
exit "$status"

