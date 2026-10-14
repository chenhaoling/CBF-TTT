#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/ctj/cbf_ttt_joint_exp_20260927}
PYTHON_BIN=${PYTHON_BIN:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
CONFIRMATION_ROOT=${CONFIRMATION_ROOT:-/home/ctj/cbf_ttt_qwen_discrete_cohort_20261015}
CONFIRMATION_CHAIN_ROOT=${CONFIRMATION_CHAIN_ROOT:-${CONFIRMATION_ROOT}.chain}
OOD_ROOT=${OOD_ROOT:-/home/ctj/cbf_ttt_qwen_discrete_cohort_ood_20261015}
CHAIN_ROOT=${CHAIN_ROOT:-${OOD_ROOT}.chain}
POLL_SECONDS=${POLL_SECONDS:-30}

test ! -e "$OOD_ROOT"
test ! -e "$CHAIN_ROOT"
mkdir -p "$CHAIN_ROOT"
trap 'code=$?; echo "$code" > "$CHAIN_ROOT/runner_exit_code.txt"' EXIT
date --iso-8601=seconds > "$CHAIN_ROOT/started_at.txt"
git -C "$REPO" rev-parse HEAD > "$CHAIN_ROOT/code_commit.txt"

# The producer chain writes this status only after its independent audit closes.
while [[ ! -f "$CONFIRMATION_CHAIN_ROOT/runner_exit_code.txt" ]]; do
  sleep "$POLL_SECONDS"
done
test "$(cat "$CONFIRMATION_CHAIN_ROOT/runner_exit_code.txt")" = "0"
test "$(cat "$CONFIRMATION_CHAIN_ROOT/terminal_status.txt")" = \
  "completed_sparse_to_discrete_confirmation"

"$PYTHON_BIN" - "$CONFIRMATION_ROOT/execution_audit.json" \
  > "$CHAIN_ROOT/confirmation_release.json" <<'PY'
import json
import sys

audit = json.load(open(sys.argv[1]))
checks = {
    "protocol": audit.get("protocol")
    == "audit_qwen_discrete_cohort_confirmation_v1",
    "audit_passed": audit.get("audit_passed") is True,
    "test_scored": audit.get("test_scored") is True,
    "test_gate_passed": audit.get("test_gate", {}).get("passed") is True,
}
result = {"checks": checks, "passed": all(checks.values())}
print(json.dumps(result, indent=2))
if not result["passed"]:
    raise SystemExit(1)
PY

cd "$REPO"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} \
  ROOT="$OOD_ROOT" REPO="$REPO" PYTHON_BIN="$PYTHON_BIN" \
  CONFIRMATION_ROOT="$CONFIRMATION_ROOT" \
  CONFIRMATION_AUDIT="$CONFIRMATION_ROOT/execution_audit.json" \
  bash scripts/run_qwen_discrete_cohort_ood_chain.sh \
  > "$CHAIN_ROOT/ood_chain.log" 2>&1

test "$(cat "$OOD_ROOT/chain_exit_code.txt")" = "0"
echo completed_discrete_confirmation_to_ood > "$CHAIN_ROOT/terminal_status.txt"
date --iso-8601=seconds > "$CHAIN_ROOT/completed_at.txt"
