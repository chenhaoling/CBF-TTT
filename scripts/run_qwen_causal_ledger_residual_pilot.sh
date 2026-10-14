#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/cbf_ttt_qwen_causal_ledger_pilot_20261015}
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
status=failed_qwen_causal_ledger_residual_pilot
code=1
finish() {
  echo "$status" > "$ROOT/terminal_status.txt"
  echo "$code" > "$ROOT/runner_exit_code.txt"
  date --iso-8601=seconds > "$ROOT/completed_at.txt"
}
trap finish EXIT

"$PYTHON_BIN" -m unittest tests.test_qwen_causal_ledger_residual \
  > "$ROOT/tests.log" 2>&1
"$PYTHON_BIN" -m tasks.build_qwen_causal_ledger_residual \
  --output "$ROOT/data" > "$ROOT/build.log" 2>&1

CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" -m tasks.pilot_qwen_causal_ledger_residual collect \
  --data-root "$ROOT/data" --model "$MODEL" --tokenizer "$TOKENIZER" \
  --shard 0 --num-shards 2 --labels-output "$ROOT/labels_shard0.jsonl" \
  --content-output "$ROOT/content_shard0.jsonl" \
  --resource-output "$ROOT/resource_shard0.json" \
  > "$ROOT/shard0.log" 2>&1 &
pid0=$!
CUDA_VISIBLE_DEVICES=1 "$PYTHON_BIN" -m tasks.pilot_qwen_causal_ledger_residual collect \
  --data-root "$ROOT/data" --model "$MODEL" --tokenizer "$TOKENIZER" \
  --shard 1 --num-shards 2 --labels-output "$ROOT/labels_shard1.jsonl" \
  --content-output "$ROOT/content_shard1.jsonl" \
  --resource-output "$ROOT/resource_shard1.json" \
  > "$ROOT/shard1.log" 2>&1 &
pid1=$!
wait "$pid0"
wait "$pid1"

"$PYTHON_BIN" -m tasks.pilot_qwen_causal_ledger_residual summarize \
  --root "$ROOT" --output "$ROOT/summary.json" > "$ROOT/summarize.log" 2>&1
status=completed_qwen_causal_ledger_residual_pilot
code=0
