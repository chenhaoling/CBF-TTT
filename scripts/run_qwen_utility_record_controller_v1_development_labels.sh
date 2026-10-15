#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/cbf_ttt_utility_record_controller_v1_development_labels_20261016}
DATA_ROOT=${DATA_ROOT:-/home/ctj/cbf_ttt_utility_record_controller_v1_development_data_20261016}
PYTHON_BIN=${PYTHON_BIN:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
MODEL=${MODEL:-/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt}
TOKENIZER=${TOKENIZER:-/home/ctj/models/Qwen3-4B}

if [[ -e "$ROOT" ]]; then
  echo "Refusing to overwrite $ROOT" >&2
  exit 2
fi
mkdir -p "$ROOT"
git rev-parse HEAD > "$ROOT/code_commit.txt"
date --iso-8601=seconds > "$ROOT/started_at.txt"
status=failed_qwen_utility_record_controller_v1_development_labels
code=1
finish() {
  echo "$status" > "$ROOT/terminal_status.txt"
  echo "$code" > "$ROOT/runner_exit_code.txt"
  date --iso-8601=seconds > "$ROOT/completed_at.txt"
}
trap finish EXIT

pids=()
for shard in 0 1; do
  CUDA_VISIBLE_DEVICES=$shard "$PYTHON_BIN" \
    -m tasks.label_qwen_utility_record_controller_v1_development collect \
    --data-root "$DATA_ROOT" --model "$MODEL" --tokenizer "$TOKENIZER" \
    --shard "$shard" --num-shards 2 \
    --output "$ROOT/labels_shard${shard}.jsonl" \
    --resource-output "$ROOT/resource_shard${shard}.json" \
    > "$ROOT/collect_shard${shard}.log" 2>&1 &
  pids+=("$!")
done
wait_code=0
for pid in "${pids[@]}"; do
  wait "$pid" || wait_code=$?
done
if [[ "$wait_code" -ne 0 ]]; then
  exit "$wait_code"
fi

"$PYTHON_BIN" -m tasks.label_qwen_utility_record_controller_v1_development summarize \
  --root "$ROOT" --data-root "$DATA_ROOT" --num-shards 2 \
  --output "$ROOT/summary.json" > "$ROOT/summarize.log" 2>&1
"$PYTHON_BIN" -m scripts.audit_qwen_utility_record_controller_v1_development_labels \
  --label-root "$ROOT" --data-root "$DATA_ROOT" \
  --output "$ROOT/development_labels_audit.json" > "$ROOT/audit.log" 2>&1

status=completed_qwen_utility_record_controller_v1_development_labels
code=0
