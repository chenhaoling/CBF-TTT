#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/cbf_ttt_docred_relation_space_projection_20261015}
DATA_ROOT=${DATA_ROOT:-/home/ctj/cbf_ttt_docred_relation_space_data_20261015_retry2}
EMBEDDING_ROOT=${EMBEDDING_ROOT:-/home/ctj/cbf_ttt_docred_relation_space_embeddings_20261015}
REPO=${REPO:-/home/ctj/cbf_ttt_joint_exp_20260927}
PYTHON_BIN=${PYTHON_BIN:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}

test ! -e "$ROOT"
mkdir -p "$ROOT"
trap 'code=$?; echo "$code" > "$ROOT/runner_exit_code.txt"' EXIT
date --iso-8601=seconds > "$ROOT/started_at.txt"
git -C "$REPO" rev-parse HEAD > "$ROOT/code_commit.txt"
cd "$REPO"
export PYTHONDONTWRITEBYTECODE=1
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" \
  -m tasks.train_docred_relation_space_projection \
  --data-root "$DATA_ROOT" --embedding-root "$EMBEDDING_ROOT" \
  --output "$ROOT/result" --device cuda:0 > "$ROOT/train.log" 2>&1
echo completed_docred_relation_space_projection > "$ROOT/terminal_status.txt"
date --iso-8601=seconds > "$ROOT/completed_at.txt"
