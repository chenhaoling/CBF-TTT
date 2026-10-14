#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/cbf_ttt_v19_nested_forgetting_policy}
REPO=${REPO:-/home/ctj/cbf_ttt_joint_exp_20260927}
PYTHON=${PYTHON:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
EXAMPLES=${EXAMPLES:-/home/ctj/cbf_ttt_v19_nested_policy_examples_20261014}

cd "$REPO"
test ! -e "$ROOT"
mkdir -p "$ROOT"
trap 'code=$?; echo "$code" > "$ROOT/runner_exit_code.txt"' EXIT
date --iso-8601=seconds > "$ROOT/started_at.txt"
git rev-parse HEAD > "$ROOT/code_commit.txt"
export PYTHONDONTWRITEBYTECODE=1
"$PYTHON" -m unittest \
  tests.test_build_v19_nested_policy_examples \
  tests.test_train_v19_nested_forgetting_policy
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m tasks.train_v19_nested_forgetting_policy \
  --examples-root "$EXAMPLES" --output "$ROOT/result" \
  > "$ROOT/train.log" 2>&1
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m scripts.audit_v19_nested_forgetting_policy \
  --examples-root "$EXAMPLES" --policy-root "$ROOT/result" \
  --output "$ROOT/audit.json" > "$ROOT/audit.log" 2>&1
echo completed_v19_nested_oof_forgetting_policy > "$ROOT/terminal_status.txt"
date --iso-8601=seconds > "$ROOT/completed_at.txt"
cat "$ROOT/result/summary.json"
