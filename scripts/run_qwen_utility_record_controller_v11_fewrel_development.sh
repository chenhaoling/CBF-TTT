#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/cbf_ttt_utility_record_controller_v11_fewrel_development_20261016}
PYTHON_BIN=${PYTHON_BIN:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
SOURCE_ROOT=${SOURCE_ROOT:-/home/ctj/data/fewrel_official_20261015}
TT_DEV=${TT_DEV:-/home/ctj/cbf_ttt_utility_record_controller_v2_templama_development_data_20261016_retry2/development_episodes.jsonl}
LAMA_DEV=${LAMA_DEV:-/home/ctj/cbf_ttt_utility_record_controller_v1_development_data_20261016/development_episodes.jsonl}
V10_TRAIN=${V10_TRAIN:-/home/ctj/cbf_ttt_utility_record_controller_v10_fewrel_corpus_20261016_retry1/data/formal_episodes.jsonl}

if [[ -e "$ROOT" ]]; then echo "Refusing to overwrite $ROOT" >&2; exit 2; fi
mkdir -p "$ROOT"
git rev-parse HEAD > "$ROOT/code_commit.txt"
date --iso-8601=seconds > "$ROOT/started_at.txt"
status=failed_qwen_utility_record_controller_v11_fewrel_development
code=1
finish() {
  echo "$status" > "$ROOT/terminal_status.txt"
  echo "$code" > "$ROOT/runner_exit_code.txt"
  date --iso-8601=seconds > "$ROOT/completed_at.txt"
}
trap finish EXIT

"$PYTHON_BIN" -m tasks.build_qwen_utility_record_controller_v11_fewrel_development \
  --source-root "$SOURCE_ROOT" \
  --exclusion-episodes "$TT_DEV" --exclusion-episodes "$LAMA_DEV" \
  --exclusion-episodes "$V10_TRAIN" \
  --output-root "$ROOT/data" > "$ROOT/build.log" 2>&1
"$PYTHON_BIN" -m scripts.audit_qwen_utility_record_controller_v11_fewrel_development \
  --data-root "$ROOT/data" --source-root "$SOURCE_ROOT" \
  --exclusion-episodes "$TT_DEV" --exclusion-episodes "$LAMA_DEV" \
  --exclusion-episodes "$V10_TRAIN" \
  --output "$ROOT/data_audit.json" > "$ROOT/audit.log" 2>&1

status=completed_qwen_utility_record_controller_v11_fewrel_development
code=0
trap - EXIT
finish
