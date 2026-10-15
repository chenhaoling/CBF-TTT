#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/cbf_ttt_qwen_natural_forgetting_factor_gate_v2_20261015}
REPO=${REPO:-/home/ctj/cbf_ttt_joint_exp_20260927}
PYTHON_BIN=${PYTHON_BIN:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
UTILITY_ROOT=${UTILITY_ROOT:-/home/ctj/cbf_ttt_qwen_natural_forgetting_utility_formal_v1_20261015}
V1_ROOT=${V1_ROOT:-/home/ctj/cbf_ttt_qwen_natural_forgetting_barrier_q_v1_20261015}

if [[ -e "$ROOT" ]]; then
  echo "Refusing to overwrite $ROOT" >&2
  exit 2
fi
mkdir -p "$ROOT"
cd "$REPO"
git rev-parse HEAD > "$ROOT/code_commit.txt"
date --iso-8601=seconds > "$ROOT/started_at.txt"
status=failed_qwen_natural_forgetting_factor_gate_v2
code=1
finish() {
  echo "$status" > "$ROOT/terminal_status.txt"
  echo "$code" > "$ROOT/runner_exit_code.txt"
  date --iso-8601=seconds > "$ROOT/completed_at.txt"
}
trap finish EXIT

"$PYTHON_BIN" -m unittest \
  tests.test_qwen_natural_forgetting_barrier_q_v1 \
  tests.test_qwen_natural_forgetting_factor_gate_v2 \
  > "$ROOT/tests.log" 2>&1

"$PYTHON_BIN" -m tasks.train_qwen_natural_forgetting_factor_gate_v2 \
  --utility-root "$UTILITY_ROOT" --v1-root "$V1_ROOT" \
  --output "$ROOT/result" > "$ROOT/train.log" 2>&1

authorized=$($PYTHON_BIN -c \
  'import json,sys; print(str(json.load(open(sys.argv[1]))["gate"]["development_authorized"]).lower())' \
  "$ROOT/result/summary.json")
if [[ "$authorized" == true ]]; then
  status=completed_qwen_natural_forgetting_factor_gate_v2_development
else
  status=completed_qwen_natural_forgetting_factor_gate_v2_calibration_stop
fi
code=0
