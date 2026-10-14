#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/cbf_ttt_qwen_discrete_cohort_ood_20261015}
REPO=${REPO:-/home/ctj/cbf_ttt_joint_exp_20260927}
PYTHON_BIN=${PYTHON_BIN:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
GEOMETRY_ROOT=${GEOMETRY_ROOT:-/home/ctj/cbf_ttt_qwen_contextual_geometry_v1/formal}
CONFIRMATION_ROOT=${CONFIRMATION_ROOT:-/home/ctj/cbf_ttt_qwen_discrete_cohort_20261015}
CONFIRMATION_AUDIT=${CONFIRMATION_AUDIT:-$CONFIRMATION_ROOT/execution_audit.json}

trap 'code=$?; if [ -d "$ROOT" ]; then echo "$code" > "$ROOT/chain_exit_code.txt"; fi' EXIT
cd "$REPO"
ROOT="$ROOT" REPO="$REPO" PYTHON_BIN="$PYTHON_BIN" \
  GEOMETRY_ROOT="$GEOMETRY_ROOT" CONFIRMATION_ROOT="$CONFIRMATION_ROOT" \
  CONFIRMATION_AUDIT="$CONFIRMATION_AUDIT" \
  bash scripts/run_qwen_discrete_cohort_ood.sh
"$PYTHON_BIN" -m scripts.audit_qwen_discrete_cohort_ood \
  --root "$ROOT" --data-root "$ROOT/data" \
  --confirmation-root "$CONFIRMATION_ROOT" \
  --confirmation-audit "$CONFIRMATION_AUDIT" \
  --geometry-root "$GEOMETRY_ROOT" \
  --output "$ROOT/execution_audit.json" \
  > "$ROOT/audit.log" 2>&1
