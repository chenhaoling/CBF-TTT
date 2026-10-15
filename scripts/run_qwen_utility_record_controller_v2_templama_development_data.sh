#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/cbf_ttt_utility_record_controller_v2_templama_development_data_20261016_retry1}
SOURCE_ROOT=${SOURCE_ROOT:-/home/ctj/cbf_ttt_templama_sources_20261016}
INSPECTION=${INSPECTION:-$SOURCE_ROOT/metadata/source_inspection_retry1.json}
PILOT_ROOT=${PILOT_ROOT:-/home/ctj/cbf_ttt_utility_record_controller_v2_templama_pilot_20261016}
PYTHON_BIN=${PYTHON_BIN:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}

if [[ -e "$ROOT" ]]; then
  echo "Refusing to overwrite $ROOT" >&2
  exit 2
fi
mkdir -p "$ROOT.runner"
git rev-parse HEAD > "$ROOT.runner/code_commit.txt"
date --iso-8601=seconds > "$ROOT.runner/started_at.txt"
status=failed_qwen_utility_record_controller_v2_templama_development_data
code=1
finish() {
  echo "$status" > "$ROOT.runner/terminal_status.txt"
  echo "$code" > "$ROOT.runner/runner_exit_code.txt"
  date --iso-8601=seconds > "$ROOT.runner/completed_at.txt"
}
trap finish EXIT

"$PYTHON_BIN" -m unittest \
  tests.test_build_qwen_utility_record_controller_v2_templama_development \
  > "$ROOT.runner/tests.log" 2>&1

"$PYTHON_BIN" \
  -m tasks.build_qwen_utility_record_controller_v2_templama_development \
  --source-root "$SOURCE_ROOT" --inspection "$INSPECTION" \
  --pilot-root "$PILOT_ROOT" --output-root "$ROOT" \
  > "$ROOT.runner/build.log" 2>&1

"$PYTHON_BIN" \
  -m scripts.audit_qwen_utility_record_controller_v2_templama_development_data \
  --data-root "$ROOT" --source-root "$SOURCE_ROOT" --inspection "$INSPECTION" \
  --pilot-root "$PILOT_ROOT" --output "$ROOT/data_audit.json" \
  > "$ROOT.runner/audit.log" 2>&1

cp "$ROOT.runner/code_commit.txt" "$ROOT/code_commit.txt"
cp "$ROOT.runner/started_at.txt" "$ROOT/started_at.txt"
status=completed_qwen_utility_record_controller_v2_templama_development_data
code=0
trap - EXIT
finish
cp "$ROOT.runner/terminal_status.txt" "$ROOT/terminal_status.txt"
cp "$ROOT.runner/runner_exit_code.txt" "$ROOT/runner_exit_code.txt"
cp "$ROOT.runner/completed_at.txt" "$ROOT/completed_at.txt"
