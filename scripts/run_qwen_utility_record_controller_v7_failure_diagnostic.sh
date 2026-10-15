#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/cbf_ttt_utility_record_controller_v7_failure_diagnostic_20261016}
PYTHON_BIN=${PYTHON_BIN:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
V7_ROOT=${V7_ROOT:-/home/ctj/cbf_ttt_utility_record_controller_v7_cross_source_training_20261016_retry1/training}
V7_AUDIT=${V7_AUDIT:-/home/ctj/cbf_ttt_utility_record_controller_v7_cross_source_training_20261016_retry1/training_audit.json}
TT_TRAIN_LABEL=${TT_TRAIN_LABEL:-/home/ctj/cbf_ttt_utility_record_controller_v4_templama_train_labels_20261016}
TT_TRAIN_FEATURE=${TT_TRAIN_FEATURE:-/home/ctj/cbf_ttt_utility_record_controller_v4_templama_train_features_20261016}
LAMA_TRAIN_LABEL=${LAMA_TRAIN_LABEL:-/home/ctj/cbf_ttt_utility_semantic_record_formal_labels_20261016_retry1}
LAMA_TRAIN_FEATURE=${LAMA_TRAIN_FEATURE:-/home/ctj/cbf_ttt_utility_record_controller_v1_features_20261016}
TT_DEV_DATA=${TT_DEV_DATA:-/home/ctj/cbf_ttt_utility_record_controller_v2_templama_development_data_20261016_retry2}
TT_DEV_LABEL=${TT_DEV_LABEL:-/home/ctj/cbf_ttt_utility_record_controller_v2_templama_development_labels_20261016}
TT_DEV_FEATURE=${TT_DEV_FEATURE:-/home/ctj/cbf_ttt_utility_record_controller_v2_templama_development_features_20261016}
LAMA_DEV_DATA=${LAMA_DEV_DATA:-/home/ctj/cbf_ttt_utility_record_controller_v1_development_data_20261016}
LAMA_DEV_LABEL=${LAMA_DEV_LABEL:-/home/ctj/cbf_ttt_utility_record_controller_v1_development_labels_20261016}
LAMA_DEV_FEATURE=${LAMA_DEV_FEATURE:-/home/ctj/cbf_ttt_utility_record_controller_v1_development_features_20261016}

if [[ -e "$ROOT" ]]; then
  echo "Refusing to overwrite $ROOT" >&2
  exit 2
fi
mkdir -p "$ROOT"
git rev-parse HEAD > "$ROOT/code_commit.txt"
date --iso-8601=seconds > "$ROOT/started_at.txt"
status=failed_qwen_utility_record_controller_v7_failure_diagnostic
code=1
finish() {
  echo "$status" > "$ROOT/terminal_status.txt"
  echo "$code" > "$ROOT/runner_exit_code.txt"
  date --iso-8601=seconds > "$ROOT/completed_at.txt"
}
trap finish EXIT

common=(
  --templama-train-label-root "$TT_TRAIN_LABEL"
  --templama-train-feature-root "$TT_TRAIN_FEATURE"
  --lama-train-label-root "$LAMA_TRAIN_LABEL"
  --lama-train-feature-root "$LAMA_TRAIN_FEATURE"
  --templama-dev-data-root "$TT_DEV_DATA"
  --templama-dev-label-root "$TT_DEV_LABEL"
  --templama-dev-feature-root "$TT_DEV_FEATURE"
  --lama-dev-data-root "$LAMA_DEV_DATA"
  --lama-dev-label-root "$LAMA_DEV_LABEL"
  --lama-dev-feature-root "$LAMA_DEV_FEATURE"
  --v7-training-root "$V7_ROOT"
  --v7-audit "$V7_AUDIT"
)

"$PYTHON_BIN" -m unittest \
  tests.test_diagnose_qwen_utility_record_controller_v7_cross_source \
  tests.test_train_qwen_utility_record_controller_v7_cross_source \
  > "$ROOT/tests.log" 2>&1

CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" \
  -m tasks.diagnose_qwen_utility_record_controller_v7_cross_source \
  "${common[@]}" --output "$ROOT/diagnostic.json" \
  > "$ROOT/diagnostic.log" 2>&1

CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" \
  -m scripts.audit_qwen_utility_record_controller_v7_failure_diagnostic \
  "${common[@]}" --diagnostic "$ROOT/diagnostic.json" \
  --output "$ROOT/diagnostic_audit.json" > "$ROOT/audit.log" 2>&1

status=completed_qwen_utility_record_controller_v7_failure_diagnostic
code=0
trap - EXIT
finish
