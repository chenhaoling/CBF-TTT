#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/cbf_ttt_qwen_natural_forgetting_consensus_v4_20261015}
REPO=${REPO:-/home/ctj/cbf_ttt_joint_exp_20260927}
PYTHON_BIN=${PYTHON_BIN:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
MODEL=${MODEL:-/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt}
TOKENIZER=${TOKENIZER:-/home/ctj/models/Qwen3-4B}
DATA_ROOT=${DATA_ROOT:-/home/ctj/cbf_ttt_qwen_natural_causal_memory_v1_20261015_r1/data}
V5_ROOT=${V5_ROOT:-/home/ctj/cbf_ttt_qwen_natural_scope_constrained_v5_20261015}
CONTEXT_ROOT=${CONTEXT_ROOT:-/home/ctj/cbf_ttt_zsre_contextual_readout_v1/formal}
SCOPE_ROOT=${SCOPE_ROOT:-/home/ctj/cbf_ttt_zsre_decomposed_scope_gate_v1/formal}
CROSSFIT_ROOT=${CROSSFIT_ROOT:-/home/ctj/cbf_ttt_qwen_natural_forgetting_crossfit_v4_20261015}

if [[ -e "$ROOT" ]]; then echo "Refusing to overwrite $ROOT" >&2; exit 2; fi
mkdir -p "$ROOT"; cd "$REPO"
git rev-parse HEAD > "$ROOT/code_commit.txt"
date --iso-8601=seconds > "$ROOT/started_at.txt"
status=failed_qwen_natural_forgetting_consensus_v4
code=1
finish() {
  echo "$status" > "$ROOT/terminal_status.txt"
  echo "$code" > "$ROOT/runner_exit_code.txt"
  date --iso-8601=seconds > "$ROOT/completed_at.txt"
}
trap finish EXIT

"$PYTHON_BIN" -m unittest \
  tests.test_qwen_natural_forgetting_crossfit_v4 \
  tests.test_qwen_natural_forgetting_consensus_v4 > "$ROOT/tests.log" 2>&1

for shard in 0 1; do
  CUDA_VISIBLE_DEVICES=$shard "$PYTHON_BIN" -m \
    tasks.evaluate_qwen_natural_forgetting_consensus_v4 collect \
    --root "$ROOT" --crossfit-root "$CROSSFIT_ROOT" --data-root "$DATA_ROOT" \
    --v5-root "$V5_ROOT" --model "$MODEL" --tokenizer "$TOKENIZER" \
    --context-root "$CONTEXT_ROOT" --scope-root "$SCOPE_ROOT" \
    --shard "$shard" --num-shards 2 > "$ROOT/collect_shard${shard}.log" 2>&1 &
  if [[ "$shard" == 0 ]]; then pid0=$!; else pid1=$!; fi
done
wait "$pid0"; wait "$pid1"

"$PYTHON_BIN" -m tasks.evaluate_qwen_natural_forgetting_consensus_v4 summarize \
  --root "$ROOT" --output "$ROOT/summary.json" --num-shards 2 \
  > "$ROOT/summarize.log" 2>&1
passed=$($PYTHON_BIN -c \
  'import json,sys; print(str(json.load(open(sys.argv[1]))["gate"]["test_authorized"]).lower())' \
  "$ROOT/summary.json")
if [[ "$passed" == true ]]; then
  status=completed_qwen_natural_forgetting_consensus_v4_test_release
else
  status=completed_qwen_natural_forgetting_consensus_v4_calibration_stop
fi
code=0
echo "$status" > "$ROOT/terminal_status.txt"
echo "$code" > "$ROOT/runner_exit_code.txt"

CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" -m scripts.audit_qwen_natural_forgetting_consensus_v4 \
  --root "$ROOT" --repo "$REPO" --crossfit-root "$CROSSFIT_ROOT" \
  --data-root "$DATA_ROOT" --v5-root "$V5_ROOT" --model "$MODEL" \
  --tokenizer "$TOKENIZER" --context-root "$CONTEXT_ROOT" \
  --scope-root "$SCOPE_ROOT" --output "$ROOT/execution_audit.json" \
  > "$ROOT/audit.log" 2>&1
