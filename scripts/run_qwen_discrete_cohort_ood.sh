#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/cbf_ttt_qwen_discrete_cohort_ood_20261015}
REPO=${REPO:-/home/ctj/cbf_ttt_joint_exp_20260927}
PYTHON_BIN=${PYTHON_BIN:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
MODEL=${MODEL:-/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt}
TOKENIZER=${TOKENIZER:-/home/ctj/models/Qwen3-4B}
GEOMETRY_ROOT=${GEOMETRY_ROOT:-/home/ctj/cbf_ttt_qwen_contextual_geometry_v1/formal}
CONFIRMATION_ROOT=${CONFIRMATION_ROOT:-/home/ctj/cbf_ttt_qwen_discrete_cohort_20261015}
CONFIRMATION_AUDIT=${CONFIRMATION_AUDIT:-$CONFIRMATION_ROOT/execution_audit.json}

cd "$REPO"
export PYTHONDONTWRITEBYTECODE=1
"$PYTHON_BIN" -m tasks.evaluate_qwen_discrete_cohort_ood verify \
  --confirmation-root "$CONFIRMATION_ROOT" \
  --confirmation-audit "$CONFIRMATION_AUDIT"
test ! -e "$ROOT"
mkdir -p "$ROOT"
trap 'code=$?; echo "$code" > "$ROOT/runner_exit_code.txt"' EXIT
date --iso-8601=seconds > "$ROOT/started_at.txt"
git -C "$REPO" rev-parse HEAD > "$ROOT/code_commit.txt"
"$PYTHON_BIN" -m tasks.build_qwen_discrete_cohort_ood \
  --output "$ROOT/data" --seed 202610154 --groups-per-suite 64 \
  > "$ROOT/build.log" 2>&1
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
  "$PYTHON_BIN" -m tasks.evaluate_qwen_discrete_cohort_ood evaluate \
  --confirmation-root "$CONFIRMATION_ROOT" \
  --confirmation-audit "$CONFIRMATION_AUDIT" \
  --data-root "$ROOT/data" --geometry-root "$GEOMETRY_ROOT" \
  --model "$MODEL" --tokenizer "$TOKENIZER" \
  --output "$ROOT/result" --batch-size 128 \
  > "$ROOT/evaluate.log" 2>&1
echo completed_qwen_discrete_cohort_ood > "$ROOT/terminal_status.txt"
date --iso-8601=seconds > "$ROOT/completed_at.txt"
