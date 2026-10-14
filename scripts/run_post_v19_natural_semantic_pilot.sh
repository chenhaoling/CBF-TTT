#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/cbf_ttt_post_v19_natural_semantic_pilot_20261015}
DATA_ROOT=${DATA_ROOT:-/home/ctj/cbf_ttt_post_v19_natural_semantic_data_20261015_retry1/data}
MODEL=${MODEL:-/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt}
TOKENIZER=${TOKENIZER:-/home/ctj/models/Qwen3-4B}
REPO=${REPO:-/home/ctj/cbf_ttt_joint_exp_20260927}
PYTHON_BIN=${PYTHON_BIN:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}

test ! -e "$ROOT"
mkdir -p "$ROOT"
trap 'code=$?; echo "$code" > "$ROOT/runner_exit_code.txt"' EXIT
date --iso-8601=seconds > "$ROOT/started_at.txt"
git -C "$REPO" rev-parse HEAD > "$ROOT/code_commit.txt"

cd "$REPO"
export PYTHONDONTWRITEBYTECODE=1
"$PYTHON_BIN" -m unittest \
  tests.test_post_v19_natural_semantic_headroom \
  tests.test_evaluate_post_v19_natural_semantic_headroom \
  > "$ROOT/tests.log" 2>&1

CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" \
  -m tasks.evaluate_post_v19_natural_semantic_headroom collect \
  --data-root "$DATA_ROOT" --split pilot --model "$MODEL" --tokenizer "$TOKENIZER" \
  --shard 0 --num-shards 2 --output "$ROOT/shard0.jsonl" \
  > "$ROOT/shard0.log" 2>&1 &
pid0=$!
CUDA_VISIBLE_DEVICES=1 "$PYTHON_BIN" \
  -m tasks.evaluate_post_v19_natural_semantic_headroom collect \
  --data-root "$DATA_ROOT" --split pilot --model "$MODEL" --tokenizer "$TOKENIZER" \
  --shard 1 --num-shards 2 --output "$ROOT/shard1.jsonl" \
  > "$ROOT/shard1.log" 2>&1 &
pid1=$!
wait "$pid0"
wait "$pid1"

"$PYTHON_BIN" -m tasks.evaluate_post_v19_natural_semantic_headroom summarize \
  --data-root "$DATA_ROOT" --split pilot \
  --inputs "$ROOT/shard0.jsonl" "$ROOT/shard1.jsonl" \
  --output "$ROOT/summary.json" > "$ROOT/summary.log" 2>&1
"$PYTHON_BIN" -m scripts.audit_post_v19_natural_semantic_outcome \
  --data-root "$DATA_ROOT" --split pilot \
  --inputs "$ROOT/shard0.jsonl" "$ROOT/shard1.jsonl" \
  --summary-path "$ROOT/summary.json" --output "$ROOT/audit.json" \
  > "$ROOT/audit.log" 2>&1
echo completed_post_v19_natural_semantic_pilot > "$ROOT/terminal_status.txt"
date --iso-8601=seconds > "$ROOT/completed_at.txt"
