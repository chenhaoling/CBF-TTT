#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/home/ctj/cbf_ttt_utility_semantic_record_formal_data_20261016}
SOURCE_ROOT=${SOURCE_ROOT:-/home/ctj/cbf_ttt_lama_pararel_sources_20261016}
INSPECTION=${INSPECTION:-$SOURCE_ROOT/metadata/source_inspection.json}
PILOT_DATA_ROOT=${PILOT_DATA_ROOT:-/home/ctj/cbf_ttt_utility_semantic_record_pilot_data_20261016_retry1}
PILOT_LABEL_ROOT=${PILOT_LABEL_ROOT:-/home/ctj/cbf_ttt_utility_semantic_record_pilot_20261016}
PILOT_RECOVERY_ROOT=${PILOT_RECOVERY_ROOT:-/home/ctj/cbf_ttt_utility_semantic_record_pilot_20261016_summary_recovery}
PYTHON_BIN=${PYTHON_BIN:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
PILOT_AUDIT=${PILOT_AUDIT:-$PILOT_RECOVERY_ROOT/pilot_audit.json}
WORK_ROOT="${DATA_ROOT}.building"

if [[ -e "$DATA_ROOT" || -e "$WORK_ROOT" ]]; then
  echo "Refusing to overwrite a formal-data artifact" >&2
  exit 2
fi

if [[ -e "$PILOT_AUDIT" ]]; then
  "$PYTHON_BIN" -c \
    'import json,sys; d=json.load(open(sys.argv[1])); assert d["passed"] and d["next_stage"] == "formal_label_construction"' \
    "$PILOT_AUDIT"
else
  "$PYTHON_BIN" -m scripts.audit_qwen_utility_semantic_record_gate_v1_pilot \
    --original-root "$PILOT_LABEL_ROOT" \
    --recovery-root "$PILOT_RECOVERY_ROOT" \
    --data-root "$PILOT_DATA_ROOT" \
    --output "$PILOT_AUDIT"
fi

"$PYTHON_BIN" -m tasks.build_qwen_utility_semantic_record_gate_v1_formal \
  --source-root "$SOURCE_ROOT" --inspection "$INSPECTION" \
  --pilot-data-root "$PILOT_DATA_ROOT" --pilot-label-audit "$PILOT_AUDIT" \
  --output-root "$WORK_ROOT"

git rev-parse HEAD > "$WORK_ROOT/code_commit.txt"
date --iso-8601=seconds > "$WORK_ROOT/started_at.txt"
"$PYTHON_BIN" -m scripts.audit_qwen_utility_semantic_record_gate_v1_formal_data \
  --source-root "$SOURCE_ROOT" --inspection "$INSPECTION" \
  --pilot-data-root "$PILOT_DATA_ROOT" --pilot-label-audit "$PILOT_AUDIT" \
  --data-root "$WORK_ROOT" --output "$WORK_ROOT/formal_data_audit.json"
echo completed_qwen_utility_semantic_record_gate_v1_formal_data \
  > "$WORK_ROOT/terminal_status.txt"
echo 0 > "$WORK_ROOT/runner_exit_code.txt"
date --iso-8601=seconds > "$WORK_ROOT/completed_at.txt"
mv "$WORK_ROOT" "$DATA_ROOT"
