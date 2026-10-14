#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/cbf_ttt_controlled_interference_v18_formal_data}
REPO=${REPO:-/home/ctj/cbf_ttt_joint_exp_20260927}
PYTHON=${PYTHON:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
PILOT_ROOT=${PILOT_ROOT:-/home/ctj/cbf_ttt_controlled_interference_v18_qwen_pilot_20261014}
PILOT_DATA=${PILOT_DATA:-/home/ctj/cbf_ttt_controlled_interference_v18_pilot_data_tokmatch_retry1/result}

cd "$REPO"
test ! -e "$ROOT"
mkdir -p "$ROOT"
trap 'code=$?; echo "$code" > "$ROOT/runner_exit_code.txt"' EXIT
date --iso-8601=seconds > "$ROOT/started_at.txt"
git rev-parse HEAD > "$ROOT/code_commit.txt"
export PYTHONDONTWRITEBYTECODE=1
"$PYTHON" -m unittest tests.test_build_controlled_interference_v18_formal
"$PYTHON" -m tasks.build_controlled_interference_v18_formal \
  --output "$ROOT/result" --pilot-root "$PILOT_ROOT" --pilot-data-root "$PILOT_DATA" \
  > "$ROOT/build.log" 2>&1
"$PYTHON" -m scripts.audit_controlled_interference_v18_formal \
  --root "$ROOT/result" --pilot-root "$PILOT_ROOT" --pilot-data-root "$PILOT_DATA" \
  --output "$ROOT/audit.json" > "$ROOT/audit.log" 2>&1
echo completed_controlled_interference_v18_formal_data > "$ROOT/terminal_status.txt"
date --iso-8601=seconds > "$ROOT/completed_at.txt"
cat "$ROOT/audit.json"
