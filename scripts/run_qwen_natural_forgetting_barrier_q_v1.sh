#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/cbf_ttt_qwen_natural_forgetting_barrier_q_v1_20261015}
REPO=${REPO:-/home/ctj/cbf_ttt_joint_exp_20260927}
PYTHON_BIN=${PYTHON_BIN:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
MODEL=${MODEL:-/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt}
TOKENIZER=${TOKENIZER:-/home/ctj/models/Qwen3-4B}
UTILITY_ROOT=${UTILITY_ROOT:-/home/ctj/cbf_ttt_qwen_natural_forgetting_utility_formal_v1_20261015}
DATA_ROOT=${DATA_ROOT:-/home/ctj/cbf_ttt_qwen_natural_causal_memory_v1_20261015_r1/data}
V5_ROOT=${V5_ROOT:-/home/ctj/cbf_ttt_qwen_natural_scope_constrained_v5_20261015}
CONTEXT_ROOT=${CONTEXT_ROOT:-/home/ctj/cbf_ttt_zsre_contextual_readout_v1/formal}
SCOPE_ROOT=${SCOPE_ROOT:-/home/ctj/cbf_ttt_zsre_decomposed_scope_gate_v1/formal}

if [[ -e "$ROOT" ]]; then
  echo "Refusing to overwrite $ROOT" >&2
  exit 2
fi
mkdir -p "$ROOT"
cd "$REPO"
git rev-parse HEAD > "$ROOT/code_commit.txt"
date --iso-8601=seconds > "$ROOT/started_at.txt"
status=failed_qwen_natural_forgetting_barrier_q_v1
code=1
finish() {
  echo "$status" > "$ROOT/terminal_status.txt"
  echo "$code" > "$ROOT/runner_exit_code.txt"
  date --iso-8601=seconds > "$ROOT/completed_at.txt"
}
trap finish EXIT

"$PYTHON_BIN" -m unittest \
  tests.test_qwen_natural_forgetting_utility_pilot_v1 \
  tests.test_qwen_natural_forgetting_barrier_q_v1 \
  > "$ROOT/tests.log" 2>&1

CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" \
  -m tasks.train_qwen_natural_forgetting_barrier_q_v1 \
  --utility-root "$UTILITY_ROOT" --data-root "$DATA_ROOT" --v5-root "$V5_ROOT" \
  --model "$MODEL" --tokenizer "$TOKENIZER" \
  --context-root "$CONTEXT_ROOT" --scope-root "$SCOPE_ROOT" \
  --output "$ROOT/result" > "$ROOT/train.log" 2>&1

authorized=$($PYTHON_BIN -c \
  'import json,sys; print(str(json.load(open(sys.argv[1]))["gate"]["development_authorized"]).lower())' \
  "$ROOT/result/summary.json")
if [[ "$authorized" == true ]]; then
  status=completed_qwen_natural_forgetting_barrier_q_v1_development
else
  status=completed_qwen_natural_forgetting_barrier_q_v1_calibration_stop
fi
code=0
