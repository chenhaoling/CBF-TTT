#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/cbf_ttt_utility_record_controller_v6_fact_key_development_20261016}
DATA_ROOT=${DATA_ROOT:-/home/ctj/cbf_ttt_utility_record_controller_v2_templama_development_data_20261016_retry2}
LABEL_ROOT=${LABEL_ROOT:-/home/ctj/cbf_ttt_utility_record_controller_v2_templama_development_labels_20261016}
FEATURE_ROOT=${FEATURE_ROOT:-/home/ctj/cbf_ttt_utility_record_controller_v2_templama_development_features_20261016}
TRAINING_ROOT=${TRAINING_ROOT:-/home/ctj/cbf_ttt_utility_record_controller_v5_residual_training_20261016}
V5_ROOT=${V5_ROOT:-/home/ctj/cbf_ttt_utility_record_controller_v5_development_evaluation_20261016}
PYTHON_BIN=${PYTHON_BIN:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}

if [[ -e "$ROOT" ]]; then
  echo "Refusing to overwrite $ROOT" >&2
  exit 2
fi
mkdir -p "$ROOT"
git rev-parse HEAD > "$ROOT/code_commit.txt"
date --iso-8601=seconds > "$ROOT/started_at.txt"
status=failed_qwen_utility_record_controller_v6_fact_key_development
code=1
finish() {
  echo "$status" > "$ROOT/terminal_status.txt"
  echo "$code" > "$ROOT/runner_exit_code.txt"
  date --iso-8601=seconds > "$ROOT/completed_at.txt"
}
trap finish EXIT

"$PYTHON_BIN" -m unittest \
  tests.test_evaluate_qwen_utility_record_controller_v6_fact_key_development \
  > "$ROOT/tests.log" 2>&1

"$PYTHON_BIN" \
  -m tasks.evaluate_qwen_utility_record_controller_v6_fact_key_development \
  --data-root "$DATA_ROOT" --feature-root "$FEATURE_ROOT" \
  --label-root "$LABEL_ROOT" --training-root "$TRAINING_ROOT" \
  --v5-root "$V5_ROOT" --output "$ROOT/result.json" \
  > "$ROOT/evaluate.log" 2>&1

"$PYTHON_BIN" \
  -m scripts.audit_qwen_utility_record_controller_v6_fact_key_development \
  --result "$ROOT/result.json" --data-root "$DATA_ROOT" \
  --feature-root "$FEATURE_ROOT" --label-root "$LABEL_ROOT" \
  --training-root "$TRAINING_ROOT" --v5-root "$V5_ROOT" \
  --output "$ROOT/result_audit.json" > "$ROOT/audit.log" 2>&1

status=completed_qwen_utility_record_controller_v6_fact_key_development
code=0
trap - EXIT
finish
