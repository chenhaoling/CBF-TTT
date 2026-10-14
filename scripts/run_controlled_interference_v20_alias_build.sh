#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/cbf_ttt_controlled_interference_v20_alias_data}
REPO=${REPO:-/home/ctj/cbf_ttt_joint_exp_20260927}
PYTHON=${PYTHON:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
RULE_ROOT=${RULE_ROOT:-/home/ctj/cbf_ttt_v19_causal_rule_diagnostic_20261014}
V19_DATA=${V19_DATA:-/home/ctj/cbf_ttt_controlled_interference_v19_data_20261014_retry1/result}

cd "$REPO"
test ! -e "$ROOT"
mkdir -p "$ROOT"
trap 'code=$?; echo "$code" > "$ROOT/runner_exit_code.txt"' EXIT
date --iso-8601=seconds > "$ROOT/started_at.txt"
git rev-parse HEAD > "$ROOT/code_commit.txt"
export PYTHONDONTWRITEBYTECODE=1
"$PYTHON" -m unittest \
  tests.test_build_controlled_interference_v20_alias \
  tests.test_evaluate_controlled_interference_v20_alias
"$PYTHON" -m tasks.build_controlled_interference_v20_alias \
  --output "$ROOT/result" --rule-root "$RULE_ROOT" \
  > "$ROOT/build.log" 2>&1
"$PYTHON" -m scripts.audit_controlled_interference_v20_alias \
  --root "$ROOT/result" --rule-root "$RULE_ROOT" \
  --v19-data-root "$V19_DATA" --output "$ROOT/audit.json" \
  > "$ROOT/audit.log" 2>&1
echo completed_controlled_interference_v20_alias_data > "$ROOT/terminal_status.txt"
date --iso-8601=seconds > "$ROOT/completed_at.txt"
cat "$ROOT/audit.json"
