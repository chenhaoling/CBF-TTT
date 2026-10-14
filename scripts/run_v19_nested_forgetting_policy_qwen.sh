#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/cbf_ttt_v19_nested_forgetting_policy_qwen}
REPO=${REPO:-/home/ctj/cbf_ttt_joint_exp_20260927}
PYTHON=${PYTHON:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
DATA=${DATA:-/home/ctj/cbf_ttt_controlled_interference_v19_data_20261014_retry1/result}
OUTCOME=${OUTCOME:-/home/ctj/cbf_ttt_controlled_interference_v19_qwen_20261014}
RULE_ROOT=${RULE_ROOT:-/home/ctj/cbf_ttt_v19_causal_rule_diagnostic_20261014}
EXAMPLES=${EXAMPLES:-/home/ctj/cbf_ttt_v19_nested_policy_examples_20261014}
POLICY=${POLICY:-/home/ctj/cbf_ttt_v19_nested_forgetting_policy_20261014}
MODEL=${MODEL:-/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt}
TOKENIZER=${TOKENIZER:-/home/ctj/models/Qwen3-4B}

cd "$REPO"
test ! -e "$ROOT"
mkdir -p "$ROOT"
trap 'code=$?; echo "$code" > "$ROOT/runner_exit_code.txt"' EXIT
date --iso-8601=seconds > "$ROOT/started_at.txt"
git rev-parse HEAD > "$ROOT/code_commit.txt"
export PYTHONDONTWRITEBYTECODE=1
"$PYTHON" -m unittest \
  tests.test_build_v19_nested_policy_examples \
  tests.test_train_v19_nested_forgetting_policy \
  tests.test_evaluate_v19_nested_forgetting_policy

CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m tasks.evaluate_v19_nested_forgetting_policy collect \
  --data-root "$DATA" --outcome-root "$OUTCOME" --rule-root "$RULE_ROOT" \
  --examples-root "$EXAMPLES" --policy-root "$POLICY" --model "$MODEL" \
  --tokenizer "$TOKENIZER" --shard 0 --num-shards 2 --output "$ROOT/shard0.jsonl" \
  > "$ROOT/shard0.log" 2>&1 &
pid0=$!
CUDA_VISIBLE_DEVICES=1 "$PYTHON" -m tasks.evaluate_v19_nested_forgetting_policy collect \
  --data-root "$DATA" --outcome-root "$OUTCOME" --rule-root "$RULE_ROOT" \
  --examples-root "$EXAMPLES" --policy-root "$POLICY" --model "$MODEL" \
  --tokenizer "$TOKENIZER" --shard 1 --num-shards 2 --output "$ROOT/shard1.jsonl" \
  > "$ROOT/shard1.log" 2>&1 &
pid1=$!
status=0
wait "$pid0" || status=$?
wait "$pid1" || status=$?
if [[ "$status" -ne 0 ]]; then
  exit "$status"
fi

"$PYTHON" -m tasks.evaluate_v19_nested_forgetting_policy summarize \
  --data-root "$DATA" --outcome-root "$OUTCOME" --rule-root "$RULE_ROOT" \
  --examples-root "$EXAMPLES" --policy-root "$POLICY" \
  --inputs "$ROOT/shard0.jsonl" "$ROOT/shard1.jsonl" \
  --output "$ROOT/summary.json" > "$ROOT/summary.log" 2>&1
"$PYTHON" -m scripts.audit_v19_nested_forgetting_policy_outcome \
  --data-root "$DATA" --outcome-root "$OUTCOME" --rule-root "$RULE_ROOT" \
  --examples-root "$EXAMPLES" --policy-root "$POLICY" --model "$MODEL" \
  --tokenizer "$TOKENIZER" --inputs "$ROOT/shard0.jsonl" "$ROOT/shard1.jsonl" \
  --summary-path "$ROOT/summary.json" --output "$ROOT/audit.json" \
  > "$ROOT/audit.log" 2>&1
echo completed_v19_nested_forgetting_policy_qwen > "$ROOT/terminal_status.txt"
date --iso-8601=seconds > "$ROOT/completed_at.txt"
cat "$ROOT/summary.json"
