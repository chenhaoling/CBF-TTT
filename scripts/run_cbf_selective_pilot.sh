#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/home/ctj/cbf_ttt_selective_forgetting_20261008}
PYTHON=${PYTHON:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
MODEL=${MODEL:-/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt}
TOKENIZER=${TOKENIZER:-/home/ctj/models/Qwen3-4B}
PARQUET=${PARQUET:-/home/ctj/data/cbf_ttt_pilot/longcrawl64/train/0-of-256.parquet}
TRAINING_DATA=${TRAINING_DATA:-/home/ctj/data/cbf_ttt_1b/mixed_1b.jsonl}
mkdir "$ROOT"
git rev-parse HEAD > "$ROOT/code_commit.txt"
date -Is > "$ROOT/started_at.txt"
"$PYTHON" -m unittest tests.test_cbf_selective tests.test_cbf_ttt > "$ROOT/tests.log" 2>&1
"$PYTHON" -m tasks.build_cbf_selective --parquet "$PARQUET" --tokenizer "$TOKENIZER" \
  --training-data "$TRAINING_DATA" --output "$ROOT/scenes.jsonl" > "$ROOT/build.log" 2>&1
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m tasks.cbf_selective collect --model "$MODEL" \
  --data "$ROOT/scenes.jsonl" --output "$ROOT/smoke.jsonl" --smoke > "$ROOT/smoke.log" 2>&1
pids=()
for gpu in 0 1; do
  CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" -m tasks.cbf_selective collect --model "$MODEL" \
    --data "$ROOT/scenes.jsonl" --output "$ROOT/labels_${gpu}.jsonl" --shard "$gpu" --shards 2 \
    > "$ROOT/collect_${gpu}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if (( status != 0 )); then exit 1; fi
"$PYTHON" -m tasks.cbf_selective summarize --data "$ROOT/scenes.jsonl" \
  --inputs "$ROOT/labels_0.jsonl" "$ROOT/labels_1.jsonl" --output "$ROOT/summary.json" > "$ROOT/summary.log" 2>&1
date -Is > "$ROOT/completed_at.txt"
