#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/home/ctj/cbf_ttt_utility_record_controller_v6_original_confirmation_data_20261016}
SOURCE_ROOT=${SOURCE_ROOT:-/home/ctj/cbf_ttt_lama_pararel_sources_20261016}
INSPECTION=${INSPECTION:-$SOURCE_ROOT/metadata/source_inspection.json}
RELEASE_ROOT=${RELEASE_ROOT:-/home/ctj/cbf_ttt_utility_record_controller_v6_templama_test_evaluation_20261016}
PYTHON_BIN=${PYTHON_BIN:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
WORK_ROOT="${DATA_ROOT}.building"

if [[ -e "$DATA_ROOT" || -e "$WORK_ROOT" ]]; then
  echo "Refusing to overwrite original confirmation data" >&2
  exit 2
fi

"$PYTHON_BIN" -m unittest \
  tests.test_qwen_utility_record_controller_v6_original_confirmation

"$PYTHON_BIN" -m tasks.build_qwen_utility_record_controller_v6_original_confirmation \
  --source-root "$SOURCE_ROOT" --inspection "$INSPECTION" \
  --release-root "$RELEASE_ROOT" --output-root "$WORK_ROOT"
git rev-parse HEAD > "$WORK_ROOT/code_commit.txt"
date --iso-8601=seconds > "$WORK_ROOT/started_at.txt"
"$PYTHON_BIN" -m scripts.audit_qwen_utility_record_controller_v6_original_confirmation_data \
  --source-root "$SOURCE_ROOT" --inspection "$INSPECTION" \
  --release-root "$RELEASE_ROOT" --data-root "$WORK_ROOT" \
  --output "$WORK_ROOT/data_audit.json"
echo completed_qwen_utility_record_controller_v6_original_confirmation_data \
  > "$WORK_ROOT/terminal_status.txt"
echo 0 > "$WORK_ROOT/runner_exit_code.txt"
date --iso-8601=seconds > "$WORK_ROOT/completed_at.txt"
mv "$WORK_ROOT" "$DATA_ROOT"
