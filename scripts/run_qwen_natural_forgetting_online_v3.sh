#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/cbf_ttt_qwen_natural_forgetting_online_v3_20261015}
REPO=${REPO:-/home/ctj/cbf_ttt_joint_exp_20260927}
PYTHON_BIN=${PYTHON_BIN:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
MODEL=${MODEL:-/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt}
TOKENIZER=${TOKENIZER:-/home/ctj/models/Qwen3-4B}
DATA_ROOT=${DATA_ROOT:-/home/ctj/cbf_ttt_qwen_natural_causal_memory_v1_20261015_r1/data}
V5_ROOT=${V5_ROOT:-/home/ctj/cbf_ttt_qwen_natural_scope_constrained_v5_20261015}
CONTEXT_ROOT=${CONTEXT_ROOT:-/home/ctj/cbf_ttt_zsre_contextual_readout_v1/formal}
SCOPE_ROOT=${SCOPE_ROOT:-/home/ctj/cbf_ttt_zsre_decomposed_scope_gate_v1/formal}
FACTOR_ROOT=${FACTOR_ROOT:-/home/ctj/cbf_ttt_qwen_natural_forgetting_factor_gate_v2_20261015}

if [[ -e "$ROOT" ]]; then
  echo "Refusing to overwrite $ROOT" >&2
  exit 2
fi
cd "$REPO"
status=failed_qwen_natural_forgetting_online_v3
code=1
finish() {
  mkdir -p "$ROOT"
  echo "$status" > "$ROOT/terminal_status.txt"
  echo "$code" > "$ROOT/runner_exit_code.txt"
  date --iso-8601=seconds > "$ROOT/completed_at.txt"
}
trap finish EXIT

"$PYTHON_BIN" -m unittest \
  tests.test_qwen_natural_forgetting_factor_gate_v2 \
  tests.test_qwen_natural_forgetting_online_v3 > /tmp/qwen_online_v3_tests.log 2>&1

"$PYTHON_BIN" -m tasks.evaluate_qwen_natural_forgetting_online_v3 release \
  --output "$ROOT" --repo "$REPO" --factor-root "$FACTOR_ROOT" \
  --data-root "$DATA_ROOT" --v5-root "$V5_ROOT" \
  --context-root "$CONTEXT_ROOT" --scope-root "$SCOPE_ROOT"
cp /tmp/qwen_online_v3_tests.log "$ROOT/tests.log"
git rev-parse HEAD > "$ROOT/code_commit.txt"
date --iso-8601=seconds > "$ROOT/started_at.txt"

CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" -m tasks.evaluate_qwen_natural_forgetting_online_v3 collect \
  --root "$ROOT" --factor-root "$FACTOR_ROOT" --data-root "$DATA_ROOT" \
  --v5-root "$V5_ROOT" --model "$MODEL" --tokenizer "$TOKENIZER" \
  --context-root "$CONTEXT_ROOT" --scope-root "$SCOPE_ROOT" \
  --shard 0 --num-shards 2 > "$ROOT/collect_shard0.log" 2>&1 &
pid0=$!
CUDA_VISIBLE_DEVICES=1 "$PYTHON_BIN" -m tasks.evaluate_qwen_natural_forgetting_online_v3 collect \
  --root "$ROOT" --factor-root "$FACTOR_ROOT" --data-root "$DATA_ROOT" \
  --v5-root "$V5_ROOT" --model "$MODEL" --tokenizer "$TOKENIZER" \
  --context-root "$CONTEXT_ROOT" --scope-root "$SCOPE_ROOT" \
  --shard 1 --num-shards 2 > "$ROOT/collect_shard1.log" 2>&1 &
pid1=$!
wait "$pid0"
wait "$pid1"

"$PYTHON_BIN" -m tasks.evaluate_qwen_natural_forgetting_online_v3 summarize \
  --root "$ROOT" --output "$ROOT/summary.json" --num-shards 2 \
  > "$ROOT/summarize.log" 2>&1

passed=$($PYTHON_BIN -c \
  'import json,sys; print(str(json.load(open(sys.argv[1]))["gate"]["passed"]).lower())' \
  "$ROOT/summary.json")
if [[ "$passed" == true ]]; then
  status=completed_qwen_natural_forgetting_online_v3_development_pass
else
  status=completed_qwen_natural_forgetting_online_v3_development_stop
fi
echo "$status" > "$ROOT/terminal_status.txt"
echo 0 > "$ROOT/runner_exit_code.txt"

CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" -m scripts.audit_qwen_natural_forgetting_online_v3 \
  --root "$ROOT" --repo "$REPO" --factor-root "$FACTOR_ROOT" \
  --data-root "$DATA_ROOT" --v5-root "$V5_ROOT" --model "$MODEL" \
  --tokenizer "$TOKENIZER" --context-root "$CONTEXT_ROOT" \
  --scope-root "$SCOPE_ROOT" --output "$ROOT/execution_audit.json" \
  > "$ROOT/audit.log" 2>&1
code=0
