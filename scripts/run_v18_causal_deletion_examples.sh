#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/cbf_ttt_v18_causal_deletion_examples_v1}
REPO=${REPO:-/home/ctj/cbf_ttt_joint_exp_20260927}
PYTHON=${PYTHON:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
DATA=${DATA:-/home/ctj/cbf_ttt_controlled_interference_v18_formal_data_20261014_retry1/result}
FORMAL_ROOT=${FORMAL_ROOT:-/home/ctj/cbf_ttt_controlled_interference_v18_qwen_formal_20261014}
BRIDGE_ROOT=${BRIDGE_ROOT:-/home/ctj/cbf_ttt_v18_lowrank_fastmemory_bridge_20261014}
PHYSICAL_ROOT=${PHYSICAL_ROOT:-/home/ctj/cbf_ttt_v18_physical_conflict_rules_20261014}

cd "$REPO"
test ! -e "$ROOT"
mkdir -p "$ROOT"
trap 'code=$?; echo "$code" > "$ROOT/runner_exit_code.txt"' EXIT
date --iso-8601=seconds > "$ROOT/started_at.txt"
git rev-parse HEAD > "$ROOT/code_commit.txt"
export PYTHONDONTWRITEBYTECODE=1
"$PYTHON" -m unittest tests.test_build_v18_causal_deletion_examples
"$PYTHON" -m tasks.build_v18_causal_deletion_examples \
  --data-root "$DATA" --formal-root "$FORMAL_ROOT" --bridge-root "$BRIDGE_ROOT" \
  --physical-root "$PHYSICAL_ROOT" --output "$ROOT/data" > "$ROOT/build.log" 2>&1
"$PYTHON" -m scripts.audit_v18_causal_deletion_examples \
  --data-root "$DATA" --formal-root "$FORMAL_ROOT" --bridge-root "$BRIDGE_ROOT" \
  --physical-root "$PHYSICAL_ROOT" --examples-root "$ROOT/data" \
  --output "$ROOT/audit.json" > "$ROOT/audit.log" 2>&1
echo completed_v18_causal_deletion_examples > "$ROOT/terminal_status.txt"
date --iso-8601=seconds > "$ROOT/completed_at.txt"
cat "$ROOT/data/manifest.json"
