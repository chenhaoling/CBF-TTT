#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/cbf_ttt_docred_relation_space_data_20261015}
SOURCE_ROOT=${SOURCE_ROOT:-/home/ctj/data/docred_official_20261015}
FEWREL_ROOT=${FEWREL_ROOT:-/home/ctj/cbf_ttt_fewrel_selective_pairs_20261015_retry1}
REPO=${REPO:-/home/ctj/cbf_ttt_joint_exp_20260927}
PYTHON_BIN=${PYTHON_BIN:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}

test ! -e "$ROOT"
mkdir -p "$ROOT"
trap 'code=$?; echo "$code" > "$ROOT/runner_exit_code.txt"' EXIT
date --iso-8601=seconds > "$ROOT/started_at.txt"
git -C "$REPO" rev-parse HEAD > "$ROOT/code_commit.txt"
cd "$REPO"
export PYTHONDONTWRITEBYTECODE=1
"$PYTHON_BIN" -m tasks.build_docred_relation_space \
  --source-root "$SOURCE_ROOT" --fewrel-root "$FEWREL_ROOT" \
  --output "$ROOT/data" > "$ROOT/build.log" 2>&1
"$PYTHON_BIN" -m scripts.audit_docred_relation_space \
  --data-root "$ROOT/data" --source-root "$SOURCE_ROOT" \
  --fewrel-root "$FEWREL_ROOT" --output "$ROOT/audit.json" \
  > "$ROOT/audit.log" 2>&1
echo completed_docred_relation_space_data > "$ROOT/terminal_status.txt"
date --iso-8601=seconds > "$ROOT/completed_at.txt"
