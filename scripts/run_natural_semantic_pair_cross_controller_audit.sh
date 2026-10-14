#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/cbf_ttt_natural_semantic_pair_cross_controller_20261015}
PAIR_ROOT=${PAIR_ROOT:-/home/ctj/cbf_ttt_natural_semantic_relation_pairs_20261015/data}
CROSS_ROOT=${CROSS_ROOT:-/home/ctj/cbf_ttt_natural_semantic_pair_cross_embeddings_20261015}
ZERO_ROOT=${ZERO_ROOT:-/home/ctj/cbf_ttt_natural_semantic_pair_cross_zero_shot_20261015}
REPO=${REPO:-/home/ctj/cbf_ttt_joint_exp_20260927}
PYTHON_BIN=${PYTHON_BIN:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}

test ! -e "$ROOT/audit.json"
cd "$REPO"
export PYTHONDONTWRITEBYTECODE=1
"$PYTHON_BIN" -m scripts.audit_natural_semantic_pair_cross_controller \
  --controller-root "$ROOT" --pair-root "$PAIR_ROOT" \
  --cross-root "$CROSS_ROOT" --zero-root "$ZERO_ROOT" \
  --output "$ROOT/audit.json" > "$ROOT/audit.log" 2>&1
echo completed_natural_semantic_pair_cross_controller_audit \
  > "$ROOT/audit_terminal_status.txt"
