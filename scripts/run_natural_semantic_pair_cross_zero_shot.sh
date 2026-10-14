#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/cbf_ttt_natural_semantic_pair_cross_zero_shot_20261015}
PAIR_ROOT=${PAIR_ROOT:-/home/ctj/cbf_ttt_natural_semantic_relation_pairs_20261015/data}
CROSS_ROOT=${CROSS_ROOT:-/home/ctj/cbf_ttt_natural_semantic_pair_cross_embeddings_20261015}
REPO=${REPO:-/home/ctj/cbf_ttt_joint_exp_20260927}
PYTHON_BIN=${PYTHON_BIN:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}

test ! -e "$ROOT"
mkdir -p "$ROOT"
trap 'code=$?; echo "$code" > "$ROOT/runner_exit_code.txt"' EXIT
date --iso-8601=seconds > "$ROOT/started_at.txt"
git -C "$REPO" rev-parse HEAD > "$ROOT/code_commit.txt"
cd "$REPO"
export PYTHONDONTWRITEBYTECODE=1
"$PYTHON_BIN" -m unittest tests.test_natural_semantic_embedding_pairs \
  > "$ROOT/tests.log" 2>&1
"$PYTHON_BIN" -m tasks.evaluate_natural_semantic_pair_cross_zero_shot \
  --pair-root "$PAIR_ROOT" --cross-root "$CROSS_ROOT" \
  --output "$ROOT/results" > "$ROOT/evaluate.log" 2>&1
echo completed_natural_semantic_pair_cross_zero_shot > "$ROOT/terminal_status.txt"
date --iso-8601=seconds > "$ROOT/completed_at.txt"
