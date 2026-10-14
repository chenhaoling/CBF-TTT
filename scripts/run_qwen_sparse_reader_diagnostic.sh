#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/cbf_ttt_qwen_sparse_reader_20261015}
REPO=${REPO:-/home/ctj/cbf_ttt_joint_exp_20260927}
PYTHON_BIN=${PYTHON_BIN:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
MODEL=${MODEL:-/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt}
TOKENIZER=${TOKENIZER:-/home/ctj/models/Qwen3-4B}
GEOMETRY_ROOT=${GEOMETRY_ROOT:-/home/ctj/cbf_ttt_qwen_contextual_geometry_v1/formal}
DATA_ROOT=${DATA_ROOT:-/home/ctj/cbf_ttt_qwen_cohort_forgetting_20261015/data}

test ! -e "$ROOT"
mkdir -p "$ROOT"
trap 'code=$?; echo "$code" > "$ROOT/runner_exit_code.txt"' EXIT
date --iso-8601=seconds > "$ROOT/started_at.txt"
git -C "$REPO" rev-parse HEAD > "$ROOT/code_commit.txt"
cd "$REPO"
export PYTHONDONTWRITEBYTECODE=1
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
  "$PYTHON_BIN" -m tasks.diagnose_qwen_sparse_reader \
  --data-root "$DATA_ROOT" --geometry-root "$GEOMETRY_ROOT" \
  --output "$ROOT/result" --model "$MODEL" --tokenizer "$TOKENIZER" \
  --batch-size 128 > "$ROOT/run.log" 2>&1
echo completed_qwen_sparse_reader_diagnostic > "$ROOT/terminal_status.txt"
date --iso-8601=seconds > "$ROOT/completed_at.txt"

