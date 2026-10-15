#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/cbf_ttt_qwen_natural_scope_joint_ranker_v3_20261015}
PYTHON_BIN=${PYTHON_BIN:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
MODEL=${MODEL:-/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt}
TOKENIZER=${TOKENIZER:-/home/ctj/models/Qwen3-4B}
DATA_ROOT=${DATA_ROOT:-/home/ctj/cbf_ttt_qwen_natural_causal_memory_v1_20261015_r1/data}
V1_ROOT=${V1_ROOT:-/home/ctj/cbf_ttt_qwen_natural_causal_memory_v1_20261015_r1}
V2_ROOT=${V2_ROOT:-/home/ctj/cbf_ttt_qwen_natural_scope_transfer_v2_20261015}
CONTEXT_ROOT=${CONTEXT_ROOT:-/home/ctj/cbf_ttt_zsre_contextual_readout_v1/formal}
SCOPE_ROOT=${SCOPE_ROOT:-/home/ctj/cbf_ttt_zsre_decomposed_scope_gate_v1/formal}

if [[ -e "$ROOT" ]]; then
  echo "Refusing to overwrite $ROOT" >&2
  exit 2
fi
mkdir -p "$ROOT"
git rev-parse HEAD > "$ROOT/code_commit.txt"
date --iso-8601=seconds > "$ROOT/started_at.txt"
status=failed_qwen_natural_scope_joint_ranker_v3
code=1
finish() {
  echo "$status" > "$ROOT/terminal_status.txt"
  echo "$code" > "$ROOT/runner_exit_code.txt"
  date --iso-8601=seconds > "$ROOT/completed_at.txt"
}
trap finish EXIT

"$PYTHON_BIN" -m unittest \
  tests.test_qwen_natural_scope_transfer_v2 \
  tests.test_qwen_natural_scope_joint_ranker_v3 > "$ROOT/tests.log" 2>&1

CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" -m tasks.train_qwen_natural_scope_joint_ranker_v3 \
  --data-root "$DATA_ROOT" --v1-root "$V1_ROOT" --v2-root "$V2_ROOT" \
  --model "$MODEL" --tokenizer "$TOKENIZER" \
  --context-root "$CONTEXT_ROOT" --scope-root "$SCOPE_ROOT" \
  --output "$ROOT/result" > "$ROOT/train.log" 2>&1

passed=$($PYTHON_BIN -c \
  'import json,sys; print(str(json.load(open(sys.argv[1]))["gate"]["passed"]).lower())' \
  "$ROOT/result/summary.json")
if [[ "$passed" == true ]]; then
  status=completed_qwen_natural_scope_joint_ranker_v3_pass
else
  status=completed_qwen_natural_scope_joint_ranker_v3_calibration_stop
fi
code=0
