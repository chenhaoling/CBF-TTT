#!/usr/bin/env bash
set -euo pipefail

FAILED_ROOT=${FAILED_ROOT:-/home/ctj/cbf_ttt_natural_semantic_embedding_geometry_20261015}
ROOT=${ROOT:-/home/ctj/cbf_ttt_natural_semantic_embedding_geometry_20261015_retry1}
PAIR_ROOT=${PAIR_ROOT:-/home/ctj/cbf_ttt_natural_semantic_relation_pairs_20261015/data}
EMBEDDING_ROOT=${EMBEDDING_ROOT:-/home/ctj/cbf_ttt_natural_semantic_surface_embeddings_20261015_retry1_audit_recovery}
RAW_ROOT=${RAW_ROOT:-/home/ctj/cbf_ttt_natural_semantic_raw_embedding_pairs_20261015}
REPO=${REPO:-/home/ctj/cbf_ttt_joint_exp_20260927}
PYTHON_BIN=${PYTHON_BIN:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}

test ! -e "$ROOT"
test "$(cat "$FAILED_ROOT/runner_exit_code.txt")" -ne 0
mkdir -p "$ROOT"
trap 'code=$?; echo "$code" > "$ROOT/runner_exit_code.txt"' EXIT
date --iso-8601=seconds > "$ROOT/started_at.txt"
git -C "$REPO" rev-parse HEAD > "$ROOT/code_commit.txt"
sha256sum "$FAILED_ROOT/diagnose.log" > "$ROOT/failed_log.sha256"

cd "$REPO"
export PYTHONDONTWRITEBYTECODE=1
"$PYTHON_BIN" -m unittest tests.test_natural_semantic_embedding_geometry \
  > "$ROOT/tests.log" 2>&1
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" \
  -m tasks.diagnose_natural_semantic_embedding_geometry \
  --pair-root "$PAIR_ROOT" --embedding-root "$EMBEDDING_ROOT" \
  --raw-root "$RAW_ROOT" --output "$ROOT/results" \
  > "$ROOT/diagnose.log" 2>&1
echo completed_natural_semantic_embedding_geometry > "$ROOT/terminal_status.txt"
date --iso-8601=seconds > "$ROOT/completed_at.txt"
