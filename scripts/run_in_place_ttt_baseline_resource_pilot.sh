#!/usr/bin/env bash
set -euo pipefail

UPSTREAM_REPO=${UPSTREAM_REPO:?set the clean official checkout}
PYTHON_BIN=${PYTHON_BIN:?set the baseline training Python}
MODEL=${MODEL:?set the Qwen3-4B model directory}
DATA=${DATA:?set the packed pilot JSONL}
OUTPUT_ROOT=${OUTPUT_ROOT:?set a new output root}
ARM=${ARM:?set ARM to ttt or no_ttt}
SEQ_LEN=${SEQ_LEN:-8192}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-2}
PLAN=${PLAN:?set the frozen baseline plan path}
TOOL_REPO=${TOOL_REPO:?set the CBF-TTT tool checkout}
UPSTREAM_COMMIT=be2324829b0e91c8fd10a74d4b43714fde6676e1

if [[ "$ARM" != "ttt" && "$ARM" != "no_ttt" ]]; then
  echo "ARM must be ttt or no_ttt" >&2; exit 2
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

CONFIG=$TOOL_REPO/configs/baseline_reproduction/qwen3_4b_resource_${ARM}.yaml
test -s "$CONFIG"
mkdir -p "$OUTPUT_ROOT/checkpoint"

"$PYTHON_BIN" - "$UPSTREAM_REPO" "$MODEL" "$DATA" "$CONFIG" "$PLAN" \
  "$OUTPUT_ROOT/run_metadata.json" "$ARM" "$SEQ_LEN" "$GLOBAL_BATCH_SIZE" <<'PY'
import hashlib, json, pathlib, subprocess, sys
upstream, model, data, config, plan, output, arm, seq_len, batch = sys.argv[1:]
sha = lambda path: hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
metadata = {
    "protocol": "in_place_ttt_baseline_resource_v1",
    "upstream_commit": subprocess.check_output(
        ["git", "-C", upstream, "rev-parse", "HEAD"], text=True).strip(),
    "upstream_clean": not subprocess.check_output(
        ["git", "-C", upstream, "status", "--short"], text=True).strip(),
    "model": model, "data": data, "arm": arm,
    "sequence_length": int(seq_len), "global_batch_size": int(batch),
    "plan_sha256": sha(plan), "config_sha256": sha(config),
    "data_sha256": sha(data), "model_config_sha256": sha(pathlib.Path(model) / "config.json"),
}
pathlib.Path(output).write_text(json.dumps(metadata, indent=2) + "\n")
PY

printf 'timestamp,index,memory_used_mib,memory_total_mib,utilization_gpu\n' > "$OUTPUT_ROOT/gpu.csv"
(
  while [[ ! -e "$OUTPUT_ROOT/monitor.stop" ]]; do
    timestamp=$(date +%s.%N)
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits | while IFS=',' read -r index used total util; do
        printf '%s,%s,%s,%s,%s\n' "$timestamp" "${index// /}" "${used// /}" \
          "${total// /}" "${util// /}" >> "$OUTPUT_ROOT/gpu.csv"
      done
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
cleanup
trap - EXIT

"$PYTHON_BIN" "$TOOL_REPO/scripts/summarize_in_place_ttt_baseline_resource.py" \
  --root "$OUTPUT_ROOT" --output "$OUTPUT_ROOT/summary.json"
exit "$status"

