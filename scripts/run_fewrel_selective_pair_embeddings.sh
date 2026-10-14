#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/cbf_ttt_fewrel_pair_embeddings_20261015}
PAIR_ROOT=${PAIR_ROOT:-/home/ctj/cbf_ttt_fewrel_selective_pairs_20261015_retry1}
REPO=${REPO:-/home/ctj/cbf_ttt_joint_exp_20260927}
PYTHON_BIN=${PYTHON_BIN:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
MODEL=${MODEL:-/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt}
TOKENIZER=${TOKENIZER:-/home/ctj/models/Qwen3-4B}
BATCH_SIZE=${BATCH_SIZE:-16}

test ! -e "$ROOT"
mkdir -p "$ROOT/embeddings"
trap 'code=$?; echo "$code" > "$ROOT/runner_exit_code.txt"' EXIT
date --iso-8601=seconds > "$ROOT/started_at.txt"
git -C "$REPO" rev-parse HEAD > "$ROOT/code_commit.txt"
cd "$REPO"
export PYTHONDONTWRITEBYTECODE=1
for shard in 0 1; do
  CUDA_VISIBLE_DEVICES=$shard "$PYTHON_BIN" \
    -m tasks.extract_fewrel_selective_pair_embeddings \
    --pair-root "$PAIR_ROOT" --model "$MODEL" --tokenizer "$TOKENIZER" \
    --output-root "$ROOT/embeddings" --shard "$shard" --num-shards 2 \
    --batch-size "$BATCH_SIZE" > "$ROOT/shard_${shard}.log" 2>&1 &
  echo $! > "$ROOT/shard_${shard}.pid"
done
wait "$(cat "$ROOT/shard_0.pid")"
wait "$(cat "$ROOT/shard_1.pid")"
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" \
  -m scripts.audit_fewrel_selective_pair_embeddings \
  --pair-root "$PAIR_ROOT" --embedding-root "$ROOT" --model "$MODEL" \
  --tokenizer "$TOKENIZER" --batch-size "$BATCH_SIZE" \
  --output "$ROOT/audit.json" > "$ROOT/audit.log" 2>&1
echo completed_fewrel_selective_pair_embeddings > "$ROOT/terminal_status.txt"
date --iso-8601=seconds > "$ROOT/completed_at.txt"
