#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/cbf_ttt_v18_causal_deletion_policy_eval_v1}
REPO=${REPO:-/home/ctj/cbf_ttt_joint_exp_20260927}
PYTHON=${PYTHON:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
DATA=${DATA:-/home/ctj/cbf_ttt_controlled_interference_v18_formal_data_20261014_retry1/result}
FORMAL_ROOT=${FORMAL_ROOT:-/home/ctj/cbf_ttt_controlled_interference_v18_qwen_formal_20261014}
BRIDGE_ROOT=${BRIDGE_ROOT:-/home/ctj/cbf_ttt_v18_lowrank_fastmemory_bridge_20261014}
PHYSICAL_ROOT=${PHYSICAL_ROOT:-/home/ctj/cbf_ttt_v18_physical_conflict_rules_20261014}
POLICY_ROOT=${POLICY_ROOT:-/home/ctj/cbf_ttt_v18_causal_deletion_policy_20261014}
MODEL=${MODEL:-/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt}
TOKENIZER=${TOKENIZER:-/home/ctj/models/Qwen3-4B}

cd "$REPO"; test ! -e "$ROOT"; mkdir -p "$ROOT"
trap 'code=$?; echo "$code" > "$ROOT/runner_exit_code.txt"' EXIT
date --iso-8601=seconds > "$ROOT/started_at.txt"; git rev-parse HEAD > "$ROOT/code_commit.txt"
export PYTHONDONTWRITEBYTECODE=1
"$PYTHON" -m unittest tests.test_evaluate_v18_causal_deletion_policy
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m tasks.evaluate_v18_causal_deletion_policy collect \
  --data-root "$DATA" --formal-root "$FORMAL_ROOT" --bridge-root "$BRIDGE_ROOT" \
  --physical-root "$PHYSICAL_ROOT" --policy-root "$POLICY_ROOT" --model "$MODEL" \
  --tokenizer "$TOKENIZER" --shard 0 --num-shards 2 --output "$ROOT/shard0.jsonl" \
  > "$ROOT/shard0.log" 2>&1 &
pid0=$!
CUDA_VISIBLE_DEVICES=1 "$PYTHON" -m tasks.evaluate_v18_causal_deletion_policy collect \
  --data-root "$DATA" --formal-root "$FORMAL_ROOT" --bridge-root "$BRIDGE_ROOT" \
  --physical-root "$PHYSICAL_ROOT" --policy-root "$POLICY_ROOT" --model "$MODEL" \
  --tokenizer "$TOKENIZER" --shard 1 --num-shards 2 --output "$ROOT/shard1.jsonl" \
  > "$ROOT/shard1.log" 2>&1 &
pid1=$!
status=0; wait "$pid0" || status=$?; wait "$pid1" || status=$?
if [[ "$status" -ne 0 ]]; then exit "$status"; fi
"$PYTHON" -m tasks.evaluate_v18_causal_deletion_policy summarize \
  --data-root "$DATA" --formal-root "$FORMAL_ROOT" --bridge-root "$BRIDGE_ROOT" \
  --physical-root "$PHYSICAL_ROOT" --policy-root "$POLICY_ROOT" \
  --inputs "$ROOT/shard0.jsonl" "$ROOT/shard1.jsonl" --output "$ROOT/summary.json" \
  > "$ROOT/summary.log" 2>&1
"$PYTHON" -m scripts.audit_v18_causal_deletion_policy_eval \
  --data-root "$DATA" --formal-root "$FORMAL_ROOT" --bridge-root "$BRIDGE_ROOT" \
  --physical-root "$PHYSICAL_ROOT" --policy-root "$POLICY_ROOT" --model "$MODEL" \
  --tokenizer "$TOKENIZER" --inputs "$ROOT/shard0.jsonl" "$ROOT/shard1.jsonl" \
  --summary-path "$ROOT/summary.json" --output "$ROOT/audit.json" > "$ROOT/audit.log" 2>&1
echo completed_v18_causal_deletion_policy_eval > "$ROOT/terminal_status.txt"
date --iso-8601=seconds > "$ROOT/completed_at.txt"; cat "$ROOT/summary.json"
