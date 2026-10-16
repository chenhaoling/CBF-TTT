#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/ctj/cbf_ttt_joint_exp_20260927}
PYTHON_BIN=${PYTHON_BIN:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
SOURCE_ROOT=${SOURCE_ROOT:-/home/ctj/data/dart_v111_official_20261016}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/ctj/cbf_ttt_utility_record_controller_v13_dart_capacity_20261016}
V10_EPISODES=${V10_EPISODES:-/home/ctj/cbf_ttt_utility_record_controller_v10_fewrel_corpus_20261016_retry1/data/formal_episodes.jsonl}
V11_EPISODES=${V11_EPISODES:-/home/ctj/cbf_ttt_utility_record_controller_v11_fewrel_development_20261016_retry1/data/development_episodes.jsonl}

if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "Refusing to overwrite $OUTPUT_ROOT" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"
date --iso-8601=seconds > "$OUTPUT_ROOT/started_at.txt"
git -C "$REPO" rev-parse HEAD > "$OUTPUT_ROOT/code_commit.txt"

set +e
"$PYTHON_BIN" -m scripts.inspect_dart_v13_capacity \
  --source-root "$SOURCE_ROOT" \
  --exclude-episodes "$V10_EPISODES" \
  --exclude-episodes "$V11_EPISODES" \
  --output-root "$OUTPUT_ROOT/data" \
  > "$OUTPUT_ROOT/inspect.log" 2>&1
INSPECT_EXIT=$?
set -e

"$PYTHON_BIN" -m scripts.audit_dart_v13_capacity \
  --source-root "$SOURCE_ROOT" \
  --exclude-episodes "$V10_EPISODES" \
  --exclude-episodes "$V11_EPISODES" \
  --capacity-root "$OUTPUT_ROOT/data" \
  --output "$OUTPUT_ROOT/capacity_audit.json" \
  > "$OUTPUT_ROOT/audit.log" 2>&1

date --iso-8601=seconds > "$OUTPUT_ROOT/completed_at.txt"
if [[ "$INSPECT_EXIT" -eq 0 ]]; then
  echo completed_qwen_utility_record_controller_v13_dart_capacity \
    > "$OUTPUT_ROOT/terminal_status.txt"
else
  echo stopped_qwen_utility_record_controller_v13_dart_capacity_failure \
    > "$OUTPUT_ROOT/terminal_status.txt"
fi
echo "$INSPECT_EXIT" > "$OUTPUT_ROOT/runner_exit_code.txt"

cat "$OUTPUT_ROOT/inspect.log"
cat "$OUTPUT_ROOT/audit.log"
exit "$INSPECT_EXIT"
