#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/cbf_ttt_qwen_dynamic_record_pilot_20261015}
PYTHON_BIN=${PYTHON_BIN:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
MODEL=${MODEL:-/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt}
TOKENIZER=${TOKENIZER:-/home/ctj/models/Qwen3-4B}
GEOMETRY_ROOT=${GEOMETRY_ROOT:-/home/ctj/cbf_ttt_qwen_contextual_geometry_v1/formal}

if [[ -e "$ROOT" ]]; then
  echo "Refusing to overwrite $ROOT" >&2
  exit 2
fi
mkdir -p "$ROOT"
git rev-parse HEAD > "$ROOT/code_commit.txt"
date --iso-8601=seconds > "$ROOT/started_at.txt"

status=failed_qwen_dynamic_record_resource_pilot
code=1
finish() {
  echo "$status" > "$ROOT/terminal_status.txt"
  echo "$code" > "$ROOT/runner_exit_code.txt"
  date --iso-8601=seconds > "$ROOT/completed_at.txt"
}
trap finish EXIT

"$PYTHON_BIN" -m unittest tests.test_qwen_dynamic_record_forgetting \
  > "$ROOT/tests.log" 2>&1
"$PYTHON_BIN" -m tasks.build_qwen_dynamic_record_forgetting \
  --output "$ROOT/data" --groups-per-family 8 --seed 202610156 \
  > "$ROOT/build.log" 2>&1

CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" -m tasks.pilot_qwen_dynamic_record_forgetting \
  --data-root "$ROOT/data" --geometry-root "$GEOMETRY_ROOT" \
  --model "$MODEL" --tokenizer "$TOKENIZER" \
  --projection-seeds 2301,2303 --device cuda --batch-size 128 \
  --output "$ROOT/shard_gpu0" > "$ROOT/shard_gpu0.log" 2>&1 &
pid0=$!
CUDA_VISIBLE_DEVICES=1 "$PYTHON_BIN" -m tasks.pilot_qwen_dynamic_record_forgetting \
  --data-root "$ROOT/data" --geometry-root "$GEOMETRY_ROOT" \
  --model "$MODEL" --tokenizer "$TOKENIZER" \
  --projection-seeds 2302 --device cuda --batch-size 128 \
  --output "$ROOT/shard_gpu1" > "$ROOT/shard_gpu1.log" 2>&1 &
pid1=$!
wait "$pid0"
wait "$pid1"

"$PYTHON_BIN" -m scripts.summarize_qwen_dynamic_record_resource_pilot \
  --root "$ROOT" --shards shard_gpu0,shard_gpu1 \
  > "$ROOT/merge.log" 2>&1

status=completed_qwen_dynamic_record_resource_pilot
code=0
