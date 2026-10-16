#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/cbf_ttt_utility_record_controller_v11_failure_diagnostic_20261016}
TRAIN_RUN=${TRAIN_RUN:-/home/ctj/cbf_ttt_utility_record_controller_v11_relation_transfer_20261016_retry1}
PYTHON_BIN=${PYTHON_BIN:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
TT_LABEL=${TT_LABEL:-/home/ctj/cbf_ttt_utility_record_controller_v4_templama_train_labels_20261016}
TT_FEATURE=${TT_FEATURE:-/home/ctj/cbf_ttt_utility_record_controller_v4_templama_train_features_20261016}
LAMA_LABEL=${LAMA_LABEL:-/home/ctj/cbf_ttt_utility_semantic_record_formal_labels_20261016_retry1}
LAMA_FEATURE=${LAMA_FEATURE:-/home/ctj/cbf_ttt_utility_record_controller_v1_features_20261016}
FEWREL_LABEL=${FEWREL_LABEL:-/home/ctj/cbf_ttt_utility_record_controller_v10_fewrel_formal_20261016}
FEWREL_FEATURE=${FEWREL_FEATURE:-/home/ctj/cbf_ttt_utility_record_controller_v10_fewrel_features_20261016}
DEV_DATA=${DEV_DATA:-/home/ctj/cbf_ttt_utility_record_controller_v11_fewrel_development_20261016_retry1/data}
DEV_LABEL=${DEV_LABEL:-/home/ctj/cbf_ttt_utility_record_controller_v11_fewrel_development_labels_20261016}
DEV_FEATURE=${DEV_FEATURE:-/home/ctj/cbf_ttt_utility_record_controller_v11_fewrel_development_features_20261016}

if [[ -e "$ROOT" ]]; then echo "Refusing to overwrite $ROOT" >&2; exit 2; fi
mkdir -p "$ROOT"
common=(
  --templama-train-label-root "$TT_LABEL" --templama-train-feature-root "$TT_FEATURE"
  --lama-train-label-root "$LAMA_LABEL" --lama-train-feature-root "$LAMA_FEATURE"
  --fewrel-train-label-root "$FEWREL_LABEL" --fewrel-train-feature-root "$FEWREL_FEATURE"
  --development-data-root "$DEV_DATA" --development-label-root "$DEV_LABEL"
  --development-feature-root "$DEV_FEATURE" --training-root "$TRAIN_RUN/training"
  --training-audit "$TRAIN_RUN/training_audit.json"
)
"$PYTHON_BIN" -m tasks.diagnose_qwen_utility_record_controller_v11_relation_transfer \
  "${common[@]}" --output "$ROOT/diagnostic.json" > "$ROOT/diagnostic.log" 2>&1
"$PYTHON_BIN" -m scripts.audit_qwen_utility_record_controller_v11_failure_diagnostic \
  "${common[@]}" --diagnostic "$ROOT/diagnostic.json" \
  --output "$ROOT/diagnostic_audit.json" > "$ROOT/audit.log" 2>&1
echo completed_qwen_utility_record_controller_v11_failure_diagnostic > "$ROOT/terminal_status.txt"
echo 0 > "$ROOT/runner_exit_code.txt"
