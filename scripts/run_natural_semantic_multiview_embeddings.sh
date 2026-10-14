#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/cbf_ttt_natural_semantic_multiview_embeddings_20261015}
PAIR_ROOT=${PAIR_ROOT:-/home/ctj/cbf_ttt_natural_semantic_relation_pairs_20261015/data}
MODEL=${MODEL:-/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt}
TOKENIZER=${TOKENIZER:-/home/ctj/models/Qwen3-4B}
REPO=${REPO:-/home/ctj/cbf_ttt_joint_exp_20260927}
PYTHON_BIN=${PYTHON_BIN:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
BATCH_SIZE=${BATCH_SIZE:-32}

test ! -e "$ROOT"
mkdir -p "$ROOT/embeddings"
trap 'code=$?; echo "$code" > "$ROOT/runner_exit_code.txt"' EXIT
date --iso-8601=seconds > "$ROOT/started_at.txt"
git -C "$REPO" rev-parse HEAD > "$ROOT/code_commit.txt"
cd "$REPO"
export PYTHONDONTWRITEBYTECODE=1
"$PYTHON_BIN" -m unittest tests.test_natural_semantic_multiview_embeddings \
  > "$ROOT/tests.log" 2>&1
for shard in 0 1; do
  CUDA_VISIBLE_DEVICES="$shard" "$PYTHON_BIN" \
    -m tasks.extract_natural_semantic_multiview_embeddings \
    --pair-root "$PAIR_ROOT" --model "$MODEL" --tokenizer "$TOKENIZER" \
    --output-root "$ROOT/embeddings" --shard "$shard" --num-shards 2 \
    --batch-size "$BATCH_SIZE" > "$ROOT/shard_${shard}.log" 2>&1 &
  eval "pid${shard}=\$!"
done
set +e
wait "$pid0"; code0=$?
wait "$pid1"; code1=$?
set -e
echo "$code0" > "$ROOT/shard_0_exit_code.txt"
echo "$code1" > "$ROOT/shard_1_exit_code.txt"
test "$code0" -eq 0
test "$code1" -eq 0
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" \
  -m scripts.audit_natural_semantic_multiview_embeddings \
  --pair-root "$PAIR_ROOT" --embedding-root "$ROOT" --model "$MODEL" \
  --tokenizer "$TOKENIZER" --output "$ROOT/audit.json" \
  --batch-size "$BATCH_SIZE" --audit-sample-size 16 > "$ROOT/audit.log" 2>&1
echo completed_natural_semantic_multiview_embeddings > "$ROOT/terminal_status.txt"
date --iso-8601=seconds > "$ROOT/completed_at.txt"
