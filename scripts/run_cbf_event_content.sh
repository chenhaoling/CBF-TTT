#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/home/ctj/cbf_ttt_event_content_v1}
SOURCE=${SOURCE:-/home/ctj/cbf_ttt_event_writer_100step_v1}
PYTHON=${PYTHON:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
MODEL=${MODEL:-/home/ctj/models/Qwen3-4B}
mkdir "$ROOT"
git rev-parse HEAD > "$ROOT/code_commit.txt"
date -Is > "$ROOT/started_at.txt"
trap 'code=$?; if (( code != 0 )); then echo "$code" > "$ROOT/failed_exit_code.txt"; date -Is > "$ROOT/failed_at.txt"; fi' EXIT
"$PYTHON" -m unittest tests.test_cbf_event_content tests.test_cbf_event_writer tests.test_cbf_event_curriculum > "$ROOT/tests.log" 2>&1
"$PYTHON" -m tasks.train_cbf_event_content prepare --root "$ROOT" --source "$SOURCE" --model "$MODEL" > "$ROOT/prepare.log" 2>&1
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m tasks.train_cbf_event_content run --root "$ROOT" --model "$MODEL" --arm ce > "$ROOT/ce.log" 2>&1 &
ce_pid=$!
CUDA_VISIBLE_DEVICES=1 "$PYTHON" -m tasks.train_cbf_event_content run --root "$ROOT" --model "$MODEL" --arm content_pair > "$ROOT/content_pair.log" 2>&1 &
pair_pid=$!
status=0
wait "$ce_pid" || status=1
wait "$pair_pid" || status=1
if (( status != 0 )); then exit 1; fi
"$PYTHON" -m tasks.train_cbf_event_content summarize --root "$ROOT" > "$ROOT/summary.log" 2>&1
date -Is > "$ROOT/completed_at.txt"
"$PYTHON" -m scripts.audit_cbf_event_content --root "$ROOT" --model "$MODEL" > "$ROOT/audit.log" 2>&1
printf '%s\n' completed_content_training > "$ROOT/terminal_status.txt"
