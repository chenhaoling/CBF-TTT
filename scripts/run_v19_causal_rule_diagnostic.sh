#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/cbf_ttt_v19_causal_rule_diagnostic}
REPO=${REPO:-/home/ctj/cbf_ttt_joint_exp_20260927}
PYTHON=${PYTHON:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
DATA=${DATA:-/home/ctj/cbf_ttt_controlled_interference_v19_data_20261014_retry1/result}
OUTCOME=${OUTCOME:-/home/ctj/cbf_ttt_controlled_interference_v19_qwen_20261014}

cd "$REPO"
test ! -e "$ROOT"
mkdir -p "$ROOT"
trap 'code=$?; echo "$code" > "$ROOT/runner_exit_code.txt"' EXIT
date --iso-8601=seconds > "$ROOT/started_at.txt"
git rev-parse HEAD > "$ROOT/code_commit.txt"
export PYTHONDONTWRITEBYTECODE=1
"$PYTHON" -m unittest tests.test_analyze_v19_causal_rules
"$PYTHON" -m tasks.analyze_v19_causal_rules \
  --data-root "$DATA" --outcome-root "$OUTCOME" \
  --inputs "$OUTCOME/shard0.jsonl" "$OUTCOME/shard1.jsonl" \
  --output "$ROOT/summary.json" > "$ROOT/run.log" 2>&1
"$PYTHON" -m scripts.audit_v19_causal_rules \
  --data-root "$DATA" --outcome-root "$OUTCOME" \
  --inputs "$OUTCOME/shard0.jsonl" "$OUTCOME/shard1.jsonl" \
  --summary-path "$ROOT/summary.json" --output "$ROOT/audit.json" \
  > "$ROOT/audit.log" 2>&1
echo completed_v19_causal_rule_diagnostic > "$ROOT/terminal_status.txt"
date --iso-8601=seconds > "$ROOT/completed_at.txt"
cat "$ROOT/summary.json"
