#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/cbf_ttt_qwen_natural_forgetting_safe_delete_dagger_v6_20261015}
REPO=${REPO:-/home/ctj/cbf_ttt_joint_exp_20260927}
PYTHON_BIN=${PYTHON_BIN:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
MODEL=${MODEL:-/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt}
TOKENIZER=${TOKENIZER:-/home/ctj/models/Qwen3-4B}
DATA_ROOT=${DATA_ROOT:-/home/ctj/cbf_ttt_qwen_natural_causal_memory_v1_20261015_r1/data}
V5_ROOT=${V5_ROOT:-/home/ctj/cbf_ttt_qwen_natural_scope_constrained_v5_20261015}
CONTEXT_ROOT=${CONTEXT_ROOT:-/home/ctj/cbf_ttt_zsre_contextual_readout_v1/formal}
SCOPE_ROOT=${SCOPE_ROOT:-/home/ctj/cbf_ttt_zsre_decomposed_scope_gate_v1/formal}
SAFE_DELETE_ROOT=${SAFE_DELETE_ROOT:-/home/ctj/cbf_ttt_qwen_natural_forgetting_safe_delete_v5_20261015_rerun1}

if [[ -e "$ROOT" ]]; then echo "Refusing to overwrite $ROOT" >&2; exit 2; fi
mkdir -p "$ROOT"; cd "$REPO"
git rev-parse HEAD > "$ROOT/code_commit.txt"
date --iso-8601=seconds > "$ROOT/started_at.txt"
status=failed_qwen_natural_forgetting_safe_delete_dagger_v6
code=1
finish() {
  echo "$status" > "$ROOT/terminal_status.txt"
  echo "$code" > "$ROOT/runner_exit_code.txt"
  date --iso-8601=seconds > "$ROOT/completed_at.txt"
}
trap finish EXIT

"$PYTHON_BIN" -m unittest -v \
  tests.test_qwen_natural_forgetting_safe_delete_v5 \
  tests.test_qwen_natural_forgetting_safe_delete_dagger_v6 \
  > "$ROOT/tests.log" 2>&1

for shard in 0 1; do
  CUDA_VISIBLE_DEVICES=$shard "$PYTHON_BIN" -m \
    tasks.train_qwen_natural_forgetting_safe_delete_dagger_v6 collect \
    --root "$ROOT" --safe-delete-root "$SAFE_DELETE_ROOT" --data-root "$DATA_ROOT" \
    --v5-root "$V5_ROOT" --model "$MODEL" --tokenizer "$TOKENIZER" \
    --context-root "$CONTEXT_ROOT" --scope-root "$SCOPE_ROOT" \
    --shard "$shard" --num-shards 2 > "$ROOT/collect_shard${shard}.log" 2>&1 &
  if [[ "$shard" == 0 ]]; then pid0=$!; else pid1=$!; fi
done
wait "$pid0"; wait "$pid1"

"$PYTHON_BIN" -m tasks.train_qwen_natural_forgetting_safe_delete_dagger_v6 summarize \
  --root "$ROOT" --safe-delete-root "$SAFE_DELETE_ROOT" \
  --output "$ROOT/dagger_summary.json" > "$ROOT/summarize.log" 2>&1
"$PYTHON_BIN" -m scripts.audit_qwen_natural_forgetting_safe_delete_dagger_v6 collect \
  --root "$ROOT" --safe-delete-root "$SAFE_DELETE_ROOT" --data-root "$DATA_ROOT" \
  --v5-root "$V5_ROOT" --context-root "$CONTEXT_ROOT" --scope-root "$SCOPE_ROOT" \
  --output "$ROOT/dagger_execution_audit.json" > "$ROOT/dagger_audit.log" 2>&1

"$PYTHON_BIN" -m tasks.train_qwen_natural_forgetting_safe_delete_dagger_v6 train \
  --safe-delete-root "$SAFE_DELETE_ROOT" --dagger-root "$ROOT" \
  --output "$ROOT/result" > "$ROOT/train.log" 2>&1
"$PYTHON_BIN" -m scripts.audit_qwen_natural_forgetting_safe_delete_dagger_v6 train \
  --root "$ROOT" --safe-delete-root "$SAFE_DELETE_ROOT" \
  --output "$ROOT/execution_audit.json" > "$ROOT/audit.log" 2>&1

passed=$($PYTHON_BIN -c \
  'import json,sys; print(str(json.load(open(sys.argv[1]))["gate"]["online_calibration_authorized"]).lower())' \
  "$ROOT/result/summary.json")
if [[ "$passed" == true ]]; then
  status=completed_qwen_natural_forgetting_safe_delete_dagger_v6_offline_pass
  code=0
else
  status=completed_qwen_natural_forgetting_safe_delete_dagger_v6_offline_stop
  code=1
fi
