#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/ctj/cbf_ttt_joint_exp_20260927}
PYTHON_BIN=${PYTHON_BIN:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
PREDECESSOR_ROOT=${PREDECESSOR_ROOT:-/home/ctj/cbf_ttt_qwen_sparse_cohort_20261015}
TARGET_ROOT=${TARGET_ROOT:-/home/ctj/cbf_ttt_qwen_discrete_cohort_20261015}
READER_AUDIT=${READER_AUDIT:-/home/ctj/cbf_ttt_qwen_sparse_reader_20261015_retry2/execution_audit.json}
CHAIN_ROOT=${CHAIN_ROOT:-${TARGET_ROOT}.chain}
POLL_SECONDS=${POLL_SECONDS:-30}

test ! -e "$TARGET_ROOT"
mkdir -p "$CHAIN_ROOT"
trap 'code=$?; echo "$code" > "$CHAIN_ROOT/runner_exit_code.txt"' EXIT
date --iso-8601=seconds > "$CHAIN_ROOT/started_at.txt"
git -C "$REPO" rev-parse HEAD > "$CHAIN_ROOT/code_commit.txt"

# The producer writes its exit code from an EXIT trap, after all result files close.
while [[ ! -f "$PREDECESSOR_ROOT/runner_exit_code.txt" ]]; do
  sleep "$POLL_SECONDS"
done
test "$(cat "$PREDECESSOR_ROOT/runner_exit_code.txt")" = "0"
test "$(cat "$PREDECESSOR_ROOT/terminal_status.txt")" = \
  "completed_qwen_sparse_cohort_forgetting"

cd "$REPO"
export PYTHONDONTWRITEBYTECODE=1
"$PYTHON_BIN" -m scripts.audit_qwen_sparse_cohort_forgetting \
  --root "$PREDECESSOR_ROOT" \
  --data-root "$PREDECESSOR_ROOT/data" \
  --reader-audit "$READER_AUDIT" \
  --output "$PREDECESSOR_ROOT/execution_audit.json" \
  > "$CHAIN_ROOT/predecessor_audit.log" 2>&1

# Recompute the frozen conditional prerequisite from the independent audit.
"$PYTHON_BIN" - "$PREDECESSOR_ROOT/execution_audit.json" \
  > "$CHAIN_ROOT/prerequisite.json" <<'PY'
import json
import sys

from scripts.audit_qwen_discrete_cohort_confirmation import prerequisites

path = sys.argv[1]
audit = json.load(open(path))
checks = prerequisites(audit)
result = {
    "predecessor_audit_passed": audit.get("audit_passed") is True,
    "predecessor_test_sealed": audit.get("test_scored") is False,
    "checks": checks,
    "passed": (
        audit.get("audit_passed") is True
        and audit.get("test_scored") is False
        and all(checks.values())
    ),
}
print(json.dumps(result, indent=2))
if not result["passed"]:
    raise SystemExit(1)
PY

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} \
  ROOT="$TARGET_ROOT" REPO="$REPO" PYTHON_BIN="$PYTHON_BIN" \
  READER_AUDIT="$READER_AUDIT" \
  PREDECESSOR_AUDIT="$PREDECESSOR_ROOT/execution_audit.json" \
  bash scripts/run_qwen_discrete_cohort_confirmation.sh \
  > "$CHAIN_ROOT/discrete_runner.log" 2>&1

"$PYTHON_BIN" -m scripts.audit_qwen_discrete_cohort_confirmation \
  --root "$TARGET_ROOT" \
  --data-root "$TARGET_ROOT/data" \
  --reader-audit "$READER_AUDIT" \
  --predecessor-audit "$PREDECESSOR_ROOT/execution_audit.json" \
  --output "$TARGET_ROOT/execution_audit.json" \
  > "$CHAIN_ROOT/discrete_audit.log" 2>&1

echo completed_sparse_to_discrete_confirmation > "$CHAIN_ROOT/terminal_status.txt"
date --iso-8601=seconds > "$CHAIN_ROOT/completed_at.txt"
