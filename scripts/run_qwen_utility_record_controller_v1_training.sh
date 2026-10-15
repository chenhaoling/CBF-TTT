#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/cbf_ttt_utility_record_controller_v1_training_20261016}
FEATURE_ROOT=${FEATURE_ROOT:-/home/ctj/cbf_ttt_utility_record_controller_v1_features_20261016}
PYTHON_BIN=${PYTHON_BIN:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}

if [[ -e "$ROOT" || -e "$ROOT.runner" ]]; then
  echo "Refusing to overwrite $ROOT or $ROOT.runner" >&2
  exit 2
fi
mkdir -p "$ROOT.runner"
git rev-parse HEAD > "$ROOT.runner/code_commit.txt"
date --iso-8601=seconds > "$ROOT.runner/started_at.txt"
status=failed_qwen_utility_record_controller_v1_training
code=1
finish() {
  echo "$status" > "$ROOT.runner/terminal_status.txt"
  echo "$code" > "$ROOT.runner/runner_exit_code.txt"
  date --iso-8601=seconds > "$ROOT.runner/completed_at.txt"
}
trap finish EXIT

"$PYTHON_BIN" -m unittest \
  tests.test_train_qwen_utility_record_controller_v1 \
  > "$ROOT.runner/tests.log" 2>&1

CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" \
  -m tasks.train_qwen_utility_record_controller_v1 \
  --feature-root "$FEATURE_ROOT" --output-root "$ROOT" --device cuda \
  > "$ROOT.runner/train.log" 2>&1

"$PYTHON_BIN" -m scripts.audit_qwen_utility_record_controller_v1_training \
  --training-root "$ROOT" --feature-root "$FEATURE_ROOT" \
  --output "$ROOT/training_audit.json" > "$ROOT.runner/audit.log" 2>&1

cp "$ROOT.runner/code_commit.txt" "$ROOT/code_commit.txt"
cp "$ROOT.runner/started_at.txt" "$ROOT/started_at.txt"
status=completed_qwen_utility_record_controller_v1_training
code=0
trap - EXIT
finish
cp "$ROOT.runner/terminal_status.txt" "$ROOT/terminal_status.txt"
cp "$ROOT.runner/runner_exit_code.txt" "$ROOT/runner_exit_code.txt"
cp "$ROOT.runner/completed_at.txt" "$ROOT/completed_at.txt"
