#!/usr/bin/env bash
set -euo pipefail

FAILED_ROOT=${FAILED_ROOT:-/home/ctj/cbf_ttt_v19_nested_forgetting_policy_20261014}
EXAMPLES=${EXAMPLES:-/home/ctj/cbf_ttt_v19_nested_policy_examples_20261014}
POLICY_ROOT=${POLICY_ROOT:-/home/ctj/cbf_ttt_v19_nested_forgetting_policy_20261014_retry1}
EVAL_ROOT=${EVAL_ROOT:-/home/ctj/cbf_ttt_v19_nested_forgetting_policy_qwen_20261014_retry1}
RECOVERY_ROOT=${RECOVERY_ROOT:-/home/ctj/cbf_ttt_v19_nested_policy_recovery_20261014}
RULE_ROOT=${RULE_ROOT:-/home/ctj/cbf_ttt_v19_causal_rule_diagnostic_20261014}
REPO=${REPO:-/home/ctj/cbf_ttt_joint_exp_20260927}
PYTHON=${PYTHON:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}

test ! -e "$RECOVERY_ROOT"
test ! -e "$POLICY_ROOT"
test ! -e "$EVAL_ROOT"
test "$(cat "$FAILED_ROOT/runner_exit_code.txt")" = "1"
grep -Fq "KeyError: 'inner_fold'" "$FAILED_ROOT/train.log"
test "$(cat "$EXAMPLES/runner_exit_code.txt")" = "0"
test "$(cat "$EXAMPLES/terminal_status.txt")" = "completed_v19_nested_policy_examples"
test "$(cat "$RULE_ROOT/runner_exit_code.txt")" = "0"
test "$(cat "$RULE_ROOT/terminal_status.txt")" = "completed_v19_causal_rule_diagnostic"

mkdir -p "$RECOVERY_ROOT"
trap 'code=$?; echo "$code" > "$RECOVERY_ROOT/runner_exit_code.txt"' EXIT
date --iso-8601=seconds > "$RECOVERY_ROOT/started_at.txt"
git -C "$REPO" rev-parse HEAD > "$RECOVERY_ROOT/code_commit.txt"
printf '%s\n' "$FAILED_ROOT" > "$RECOVERY_ROOT/superseded_failed_root.txt"

cd "$REPO"
export PYTHONDONTWRITEBYTECODE=1
"$PYTHON" -m unittest \
  tests.test_train_v19_nested_forgetting_policy \
  tests.test_evaluate_v19_nested_forgetting_policy \
  > "$RECOVERY_ROOT/tests.log" 2>&1

ROOT="$POLICY_ROOT" EXAMPLES="$EXAMPLES" \
  bash scripts/run_v19_nested_forgetting_policy.sh \
  > "$RECOVERY_ROOT/policy_runner.log" 2>&1
echo audited_policy_retry_launching_qwen_rescore > "$RECOVERY_ROOT/release_status.txt"
date --iso-8601=seconds > "$RECOVERY_ROOT/eval_released_at.txt"

ROOT="$EVAL_ROOT" EXAMPLES="$EXAMPLES" POLICY="$POLICY_ROOT" RULE_ROOT="$RULE_ROOT" \
  bash scripts/run_v19_nested_forgetting_policy_qwen.sh \
  > "$RECOVERY_ROOT/eval_runner.log" 2>&1

echo completed_v19_nested_policy_recovery_and_qwen_eval > "$RECOVERY_ROOT/terminal_status.txt"
date --iso-8601=seconds > "$RECOVERY_ROOT/completed_at.txt"
