#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/cbf_ttt_qwen_discrete_cohort_20261015}
REPO=${REPO:-/home/ctj/cbf_ttt_joint_exp_20260927}
PYTHON_BIN=${PYTHON_BIN:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
MODEL=${MODEL:-/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt}
TOKENIZER=${TOKENIZER:-/home/ctj/models/Qwen3-4B}
GEOMETRY_ROOT=${GEOMETRY_ROOT:-/home/ctj/cbf_ttt_qwen_contextual_geometry_v1/formal}
READER_AUDIT=${READER_AUDIT:-/home/ctj/cbf_ttt_qwen_sparse_reader_20261015_retry2/execution_audit.json}
PREDECESSOR_AUDIT=${PREDECESSOR_AUDIT:-/home/ctj/cbf_ttt_qwen_sparse_cohort_20261015/execution_audit.json}

test ! -e "$ROOT"
test -f "$READER_AUDIT"
test -f "$PREDECESSOR_AUDIT"
mkdir -p "$ROOT"
trap 'code=$?; echo "$code" > "$ROOT/runner_exit_code.txt"' EXIT
date --iso-8601=seconds > "$ROOT/started_at.txt"
git -C "$REPO" rev-parse HEAD > "$ROOT/code_commit.txt"
cd "$REPO"
export PYTHONDONTWRITEBYTECODE=1
"$PYTHON_BIN" -m tasks.build_cbf_event_curriculum \
  --output "$ROOT/data" --seed 202610153 \
  --train-groups 256 --dev-groups 64 --test-groups 128 \
  > "$ROOT/build.log" 2>&1
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
  "$PYTHON_BIN" -m tasks.train_qwen_discrete_cohort_confirmation \
  --data-root "$ROOT/data" --geometry-root "$GEOMETRY_ROOT" \
  --reader-audit "$READER_AUDIT" --predecessor-audit "$PREDECESSOR_AUDIT" \
  --output "$ROOT/result" --model "$MODEL" --tokenizer "$TOKENIZER" \
  --batch-size 128 --epochs 300 > "$ROOT/train.log" 2>&1
echo completed_qwen_discrete_cohort_confirmation > "$ROOT/terminal_status.txt"
date --iso-8601=seconds > "$ROOT/completed_at.txt"

