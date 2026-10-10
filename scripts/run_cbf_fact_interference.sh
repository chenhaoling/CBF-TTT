#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/home/ctj/cbf_ttt_fact_interference_v1}
SOURCE=${SOURCE:-/home/ctj/cbf_ttt_event_content_v1}
PYTHON=${PYTHON:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
MODEL=${MODEL:-/home/ctj/models/Qwen3-4B}
mkdir "$ROOT"
git rev-parse HEAD > "$ROOT/code_commit.txt"
date -Is > "$ROOT/started_at.txt"
trap 'code=$?; if (( code != 0 )); then echo "$code" > "$ROOT/failed_exit_code.txt"; date -Is > "$ROOT/failed_at.txt"; fi' EXIT
"$PYTHON" -m unittest tests.test_cbf_fact_interference tests.test_cbf_event_content tests.test_cbf_event_writer tests.test_cbf_event_curriculum > "$ROOT/tests.log" 2>&1
"$PYTHON" -m tasks.cbf_fact_interference prepare --root "$ROOT" --source "$SOURCE" --model "$MODEL" > "$ROOT/prepare.log" 2>&1
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m tasks.cbf_fact_interference run --root "$ROOT" --model "$MODEL" --arm sequential > "$ROOT/sequential.log" 2>&1 &
seq_pid=$!
(
  CUDA_VISIBLE_DEVICES=1 "$PYTHON" -m tasks.cbf_fact_interference run --root "$ROOT" --model "$MODEL" --arm joint_context > "$ROOT/joint_context.log" 2>&1
  CUDA_VISIBLE_DEVICES=1 "$PYTHON" -m tasks.cbf_fact_interference run --root "$ROOT" --model "$MODEL" --arm mixed_context > "$ROOT/mixed_context.log" 2>&1
) &
group_pid=$!
status=0
wait "$seq_pid" || status=1
wait "$group_pid" || status=1
if (( status != 0 )); then exit 1; fi
"$PYTHON" -m tasks.cbf_fact_interference summarize --root "$ROOT" > "$ROOT/summary.log" 2>&1
date -Is > "$ROOT/completed_at.txt"
"$PYTHON" -m scripts.audit_cbf_fact_interference --root "$ROOT" --model "$MODEL" > "$ROOT/audit.log" 2>&1
printf '%s\n' completed_fact_interference > "$ROOT/terminal_status.txt"
