#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/cbf_ttt_post_v19_natural_semantic_data_20261015}
INVENTORY_ROOT=${INVENTORY_ROOT:-/home/ctj/cbf_ttt_semantic_controller_inventory_20261014/inventory}
REPO=${REPO:-/home/ctj/cbf_ttt_joint_exp_20260927}
PYTHON_BIN=${PYTHON_BIN:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}

test ! -e "$ROOT"
mkdir -p "$ROOT"
trap 'code=$?; echo "$code" > "$ROOT/runner_exit_code.txt"' EXIT
date --iso-8601=seconds > "$ROOT/started_at.txt"
git -C "$REPO" rev-parse HEAD > "$ROOT/code_commit.txt"

cd "$REPO"
export PYTHONDONTWRITEBYTECODE=1
"$PYTHON_BIN" -m unittest tests.test_post_v19_natural_semantic_headroom \
  > "$ROOT/tests.log" 2>&1
"$PYTHON_BIN" -m tasks.build_post_v19_natural_semantic_headroom \
  --inventory-root "$INVENTORY_ROOT" --output "$ROOT/data" \
  > "$ROOT/build.log" 2>&1
"$PYTHON_BIN" -m scripts.audit_post_v19_natural_semantic_headroom \
  --inventory-root "$INVENTORY_ROOT" --data-root "$ROOT/data" \
  --output "$ROOT/audit.json" > "$ROOT/audit.log" 2>&1
echo completed_post_v19_natural_semantic_data > "$ROOT/terminal_status.txt"
date --iso-8601=seconds > "$ROOT/completed_at.txt"
