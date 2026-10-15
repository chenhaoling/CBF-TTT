#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/cbf_ttt_utility_record_controller_v8_relation_posterior_features_20261016_retry1}
PYTHON_BIN=${PYTHON_BIN:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
MODEL=${MODEL:-/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt}
TOKENIZER=${TOKENIZER:-/home/ctj/models/Qwen3-4B}
TT_TRAIN_DATA=${TT_TRAIN_DATA:-/home/ctj/cbf_ttt_utility_record_controller_v4_templama_train_data_20261016}
TT_TRAIN_FEATURE=${TT_TRAIN_FEATURE:-/home/ctj/cbf_ttt_utility_record_controller_v4_templama_train_features_20261016}
LAMA_TRAIN_DATA=${LAMA_TRAIN_DATA:-/home/ctj/cbf_ttt_utility_semantic_record_formal_data_20261016}
LAMA_TRAIN_FEATURE=${LAMA_TRAIN_FEATURE:-/home/ctj/cbf_ttt_utility_record_controller_v1_features_20261016}
TT_DEV_DATA=${TT_DEV_DATA:-/home/ctj/cbf_ttt_utility_record_controller_v2_templama_development_data_20261016_retry2}
TT_DEV_FEATURE=${TT_DEV_FEATURE:-/home/ctj/cbf_ttt_utility_record_controller_v2_templama_development_features_20261016}
LAMA_DEV_DATA=${LAMA_DEV_DATA:-/home/ctj/cbf_ttt_utility_record_controller_v1_development_data_20261016}
LAMA_DEV_FEATURE=${LAMA_DEV_FEATURE:-/home/ctj/cbf_ttt_utility_record_controller_v1_development_features_20261016}

if [[ -e "$ROOT" ]]; then
  echo "Refusing to overwrite $ROOT" >&2
  exit 2
fi
mkdir -p "$ROOT"
git rev-parse HEAD > "$ROOT/code_commit.txt"
date --iso-8601=seconds > "$ROOT/started_at.txt"
status=failed_qwen_utility_record_controller_v8_relation_posterior_features
code=1
finish() {
  echo "$status" > "$ROOT/terminal_status.txt"
  echo "$code" > "$ROOT/runner_exit_code.txt"
  date --iso-8601=seconds > "$ROOT/completed_at.txt"
}
trap finish EXIT

common=(
  --templama-train-data-root "$TT_TRAIN_DATA"
  --templama-train-feature-root "$TT_TRAIN_FEATURE"
  --lama-train-data-root "$LAMA_TRAIN_DATA"
  --lama-train-feature-root "$LAMA_TRAIN_FEATURE"
  --templama-dev-data-root "$TT_DEV_DATA"
  --templama-dev-feature-root "$TT_DEV_FEATURE"
  --lama-dev-data-root "$LAMA_DEV_DATA"
  --lama-dev-feature-root "$LAMA_DEV_FEATURE"
  --model "$MODEL" --tokenizer "$TOKENIZER"
)

"$PYTHON_BIN" -m unittest \
  tests.test_extract_qwen_utility_record_controller_v8_relation_posterior_features \
  > "$ROOT/tests.log" 2>&1

CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" \
  -m tasks.extract_qwen_utility_record_controller_v8_relation_posterior_features \
  "${common[@]}" --output-root "$ROOT/features" \
  > "$ROOT/extract.log" 2>&1

"$PYTHON_BIN" \
  -m scripts.audit_qwen_utility_record_controller_v8_relation_posterior_features \
  "${common[@]}" --feature-root "$ROOT/features" \
  --output "$ROOT/feature_audit.json" > "$ROOT/audit.log" 2>&1

status=completed_qwen_utility_record_controller_v8_relation_posterior_features
code=0
trap - EXIT
finish
