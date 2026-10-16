#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/cbf_ttt_utility_record_controller_v10_failure_diagnostic_20261016}
PYTHON_BIN=${PYTHON_BIN:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
TRAIN_RUN=${TRAIN_RUN:-/home/ctj/cbf_ttt_utility_record_controller_v10_three_source_20261016_retry1}
TT_TRAIN_LABEL=${TT_TRAIN_LABEL:-/home/ctj/cbf_ttt_utility_record_controller_v4_templama_train_labels_20261016}
TT_TRAIN_FEATURE=${TT_TRAIN_FEATURE:-/home/ctj/cbf_ttt_utility_record_controller_v4_templama_train_features_20261016}
LAMA_TRAIN_LABEL=${LAMA_TRAIN_LABEL:-/home/ctj/cbf_ttt_utility_semantic_record_formal_labels_20261016_retry1}
LAMA_TRAIN_FEATURE=${LAMA_TRAIN_FEATURE:-/home/ctj/cbf_ttt_utility_record_controller_v1_features_20261016}
FEWREL_LABEL=${FEWREL_LABEL:-/home/ctj/cbf_ttt_utility_record_controller_v10_fewrel_formal_20261016}
FEWREL_FEATURE=${FEWREL_FEATURE:-/home/ctj/cbf_ttt_utility_record_controller_v10_fewrel_features_20261016}
TT_DEV_DATA=${TT_DEV_DATA:-/home/ctj/cbf_ttt_utility_record_controller_v2_templama_development_data_20261016_retry2}
TT_DEV_LABEL=${TT_DEV_LABEL:-/home/ctj/cbf_ttt_utility_record_controller_v2_templama_development_labels_20261016}
TT_DEV_FEATURE=${TT_DEV_FEATURE:-/home/ctj/cbf_ttt_utility_record_controller_v2_templama_development_features_20261016}
LAMA_DEV_DATA=${LAMA_DEV_DATA:-/home/ctj/cbf_ttt_utility_record_controller_v1_development_data_20261016}
LAMA_DEV_LABEL=${LAMA_DEV_LABEL:-/home/ctj/cbf_ttt_utility_record_controller_v1_development_labels_20261016}
LAMA_DEV_FEATURE=${LAMA_DEV_FEATURE:-/home/ctj/cbf_ttt_utility_record_controller_v1_development_features_20261016}
V8_RUN=${V8_RUN:-/home/ctj/cbf_ttt_utility_record_controller_v8_relation_posterior_training_20261016}

if [[ -e "$ROOT" ]]; then
  echo "Refusing to overwrite $ROOT" >&2
  exit 2
fi
mkdir -p "$ROOT"
git rev-parse HEAD > "$ROOT/code_commit.txt"
date --iso-8601=seconds > "$ROOT/started_at.txt"
status=failed_qwen_utility_record_controller_v10_failure_diagnostic
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
  --fewrel-label-root "$FEWREL_LABEL"
  --fewrel-feature-root "$FEWREL_FEATURE"
  --templama-dev-data-root "$TT_DEV_DATA"
  --templama-dev-label-root "$TT_DEV_LABEL"
  --templama-dev-feature-root "$TT_DEV_FEATURE"
  --lama-dev-data-root "$LAMA_DEV_DATA"
  --lama-dev-label-root "$LAMA_DEV_LABEL"
  --lama-dev-feature-root "$LAMA_DEV_FEATURE"
  --v8-training-root "$V8_RUN/training"
  --v8-audit "$V8_RUN/training_audit.json"
  --training-root "$TRAIN_RUN/training"
  --training-audit "$TRAIN_RUN/training_audit.json"
)

"$PYTHON_BIN" -m tasks.diagnose_qwen_utility_record_controller_v10_three_source \
  "${common[@]}" --output "$ROOT/diagnostic.json" \
  > "$ROOT/diagnostic.log" 2>&1

"$PYTHON_BIN" -m scripts.audit_qwen_utility_record_controller_v10_failure_diagnostic \
  "${common[@]}" --diagnostic "$ROOT/diagnostic.json" \
  --output "$ROOT/diagnostic_audit.json" > "$ROOT/audit.log" 2>&1

status=completed_qwen_utility_record_controller_v10_failure_diagnostic
code=0
trap - EXIT
finish
