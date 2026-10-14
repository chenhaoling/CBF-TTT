#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT=${SOURCE_ROOT:-/home/ctj/cbf_ttt_natural_semantic_surface_embeddings_20261015}
ROOT=${ROOT:-/home/ctj/cbf_ttt_natural_semantic_surface_embeddings_20261015_retry1_audit_recovery}
PAIR_ROOT=${PAIR_ROOT:-/home/ctj/cbf_ttt_natural_semantic_relation_pairs_20261015/data}
MODEL=${MODEL:-/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt}
TOKENIZER=${TOKENIZER:-/home/ctj/models/Qwen3-4B}
REPO=${REPO:-/home/ctj/cbf_ttt_joint_exp_20260927}
PYTHON_BIN=${PYTHON_BIN:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}

test ! -e "$ROOT"
test "$(cat "$SOURCE_ROOT/shard_0_exit_code.txt")" -eq 0
test "$(cat "$SOURCE_ROOT/shard_1_exit_code.txt")" -eq 0
mkdir -p "$ROOT"
trap 'code=$?; echo "$code" > "$ROOT/runner_exit_code.txt"' EXIT
date --iso-8601=seconds > "$ROOT/started_at.txt"
git -C "$REPO" rev-parse HEAD > "$ROOT/code_commit.txt"
sha256sum "$SOURCE_ROOT/audit.json" > "$ROOT/source_failed_audit.sha256"
cp -a "$SOURCE_ROOT/embeddings" "$ROOT/embeddings"

cd "$REPO"
export PYTHONDONTWRITEBYTECODE=1
"$PYTHON_BIN" -m unittest tests.test_natural_semantic_surface_embeddings \
  > "$ROOT/tests.log" 2>&1
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" \
  -m scripts.audit_natural_semantic_surface_embeddings \
  --pair-root "$PAIR_ROOT" --embedding-root "$ROOT/embeddings" \
  --model "$MODEL" --tokenizer "$TOKENIZER" --output "$ROOT/audit.json" \
  --num-shards 2 --audit-sample-size 32 --extraction-batch-size 32 \
  > "$ROOT/audit.log" 2>&1
echo completed_natural_semantic_surface_embeddings > "$ROOT/terminal_status.txt"
date --iso-8601=seconds > "$ROOT/completed_at.txt"
