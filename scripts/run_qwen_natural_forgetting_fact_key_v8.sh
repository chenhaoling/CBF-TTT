#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/cbf_ttt_qwen_natural_forgetting_fact_key_v8_20261015}
REPO=${REPO:-/home/ctj/cbf_ttt_joint_exp_20260927}
PYTHON_BIN=${PYTHON_BIN:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
COUNTERFACT_SOURCE=${COUNTERFACT_SOURCE:-/home/ctj/data/cbf_ttt_eval/counterfact/counterfact.json}
V7_ROOT=${V7_ROOT:-/home/ctj/cbf_ttt_qwen_natural_forgetting_entity_key_v7_20261015}
DATA_ROOT=${DATA_ROOT:-/home/ctj/cbf_ttt_qwen_natural_causal_memory_v1_20261015_r1/data}

if [[ -e "$ROOT" ]]; then echo "Refusing to overwrite $ROOT" >&2; exit 2; fi
mkdir -p "$ROOT"; cd "$REPO"
git rev-parse HEAD > "$ROOT/code_commit.txt"
date --iso-8601=seconds > "$ROOT/started_at.txt"
status=failed_qwen_natural_forgetting_fact_key_v8
code=1
finish() {
  echo "$status" > "$ROOT/terminal_status.txt"
  echo "$code" > "$ROOT/runner_exit_code.txt"
  date --iso-8601=seconds > "$ROOT/completed_at.txt"
}
trap finish EXIT

"$PYTHON_BIN" -m unittest -v tests.test_qwen_natural_forgetting_fact_key_v8 \
  > "$ROOT/tests.log" 2>&1
"$PYTHON_BIN" -m tasks.train_qwen_natural_forgetting_fact_key_v8 train \
  --counterfact-source "$COUNTERFACT_SOURCE" --output "$ROOT/independent" \
  > "$ROOT/train.log" 2>&1
"$PYTHON_BIN" -m scripts.audit_qwen_natural_forgetting_fact_key_v8 train \
  --root "$ROOT/independent" --counterfact-source "$COUNTERFACT_SOURCE" \
  --output "$ROOT/independent/execution_audit.json" > "$ROOT/train_audit.log" 2>&1

released=$($PYTHON_BIN -c \
  'import json,sys; print(str(json.load(open(sys.argv[1]))["gate"]["zsre_transfer_authorized"]).lower())' \
  "$ROOT/independent/summary.json")
if [[ "$released" != true ]]; then
  status=completed_qwen_natural_forgetting_fact_key_v8_independent_stop
  code=1
  exit 1
fi

"$PYTHON_BIN" -m tasks.train_qwen_natural_forgetting_fact_key_v8 transfer \
  --independent-root "$ROOT/independent" --v7-root "$V7_ROOT" \
  --data-root "$DATA_ROOT" --output "$ROOT/transfer" > "$ROOT/transfer.log" 2>&1
"$PYTHON_BIN" -m scripts.audit_qwen_natural_forgetting_fact_key_v8 transfer \
  --root "$ROOT/transfer" --independent-root "$ROOT/independent" \
  --v7-root "$V7_ROOT" --data-root "$DATA_ROOT" \
  --output "$ROOT/transfer/execution_audit.json" > "$ROOT/transfer_audit.log" 2>&1

passed=$($PYTHON_BIN -c \
  'import json,sys; print(str(json.load(open(sys.argv[1]))["gate"]["online_calibration_authorized"]).lower())' \
  "$ROOT/transfer/summary.json")
if [[ "$passed" == true ]]; then
  status=completed_qwen_natural_forgetting_fact_key_v8_transfer_pass
  code=0
else
  status=completed_qwen_natural_forgetting_fact_key_v8_transfer_stop
  code=1
fi
