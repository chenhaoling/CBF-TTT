#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/ctj/cbf_ttt_joint_exp_20260927}
PYTHON=${PYTHON:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
ROOT=${ROOT:-/home/ctj/cbf_ttt_qwen_natural_forgetting_safe_delete_v5_20261015}
SOURCE_ROOT=${SOURCE_ROOT:-/home/ctj/cbf_ttt_qwen_natural_forgetting_consensus_v4_20261015}

cd "$REPO"
test ! -e "$ROOT"
mkdir -p "$ROOT"
git rev-parse HEAD > "$ROOT/code_commit.txt"

set +e
"$PYTHON" -m pytest -q tests/test_qwen_natural_forgetting_safe_delete_v5.py \
  > "$ROOT/tests.log" 2>&1
test_code=$?
set -e
if [[ $test_code -ne 0 ]]; then
  printf '%s\n' "$test_code" > "$ROOT/runner_exit_code.txt"
  printf '%s\n' failed_qwen_natural_forgetting_safe_delete_v5_tests \
    > "$ROOT/terminal_status.txt"
  exit "$test_code"
fi

"$PYTHON" -m tasks.train_qwen_natural_forgetting_safe_delete_v5 \
  --source-root "$SOURCE_ROOT" --output "$ROOT/result" \
  > "$ROOT/train.log" 2>&1
"$PYTHON" scripts/audit_qwen_natural_forgetting_safe_delete_v5.py \
  --root "$ROOT/result" --source-root "$SOURCE_ROOT" \
  --output "$ROOT/execution_audit.json" > "$ROOT/audit.log" 2>&1

"$PYTHON" - "$ROOT" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
summary = json.loads((root / "result" / "summary.json").read_text())
audit = json.loads((root / "execution_audit.json").read_text())
passed = summary["gate"]["online_calibration_authorized"] and audit["passed"]
status = ("completed_qwen_natural_forgetting_safe_delete_v5_offline_pass"
          if passed else
          "completed_qwen_natural_forgetting_safe_delete_v5_offline_stop")
(root / "terminal_status.txt").write_text(status + "\n")
(root / "runner_exit_code.txt").write_text(("0" if passed else "1") + "\n")
raise SystemExit(0 if passed else 1)
PY
