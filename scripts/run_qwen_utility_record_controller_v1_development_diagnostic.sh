#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/cbf_ttt_utility_record_controller_v1_development_diagnostic_20261016}
FEATURE_ROOT=${FEATURE_ROOT:-/home/ctj/cbf_ttt_utility_record_controller_v1_development_features_20261016}
LABEL_ROOT=${LABEL_ROOT:-/home/ctj/cbf_ttt_utility_record_controller_v1_development_labels_20261016}
TRAINING_ROOT=${TRAINING_ROOT:-/home/ctj/cbf_ttt_utility_record_controller_v1_training_20261016_retry1}
PYTHON_BIN=${PYTHON_BIN:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}

if [[ -e "$ROOT" ]]; then
  echo "Refusing to overwrite $ROOT" >&2
  exit 2
fi
mkdir -p "$ROOT"
git rev-parse HEAD > "$ROOT/code_commit.txt"
date --iso-8601=seconds > "$ROOT/started_at.txt"
status=failed_qwen_utility_record_controller_v1_development_diagnostic
code=1
finish() {
  echo "$status" > "$ROOT/terminal_status.txt"
  echo "$code" > "$ROOT/runner_exit_code.txt"
  date --iso-8601=seconds > "$ROOT/completed_at.txt"
}
trap finish EXIT

"$PYTHON_BIN" -m unittest \
  tests.test_evaluate_qwen_utility_record_controller_v1_development \
  > "$ROOT/tests.log" 2>&1

CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" \
  -m tasks.evaluate_qwen_utility_record_controller_v1_development \
  --feature-root "$FEATURE_ROOT" --label-root "$LABEL_ROOT" \
  --training-root "$TRAINING_ROOT" --output "$ROOT/result.json" \
  > "$ROOT/evaluate.log" 2>&1

CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" \
  -m scripts.audit_qwen_utility_record_controller_v1_development \
  --result "$ROOT/result.json" --feature-root "$FEATURE_ROOT" \
  --label-root "$LABEL_ROOT" --training-root "$TRAINING_ROOT" \
  --output "$ROOT/development_diagnostic_audit.json" \
  > "$ROOT/audit.log" 2>&1

status=completed_qwen_utility_record_controller_v1_development_diagnostic
code=0
