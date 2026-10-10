#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/home/ctj/cbf_ttt_event_writer_100step_v1}
SOURCE=${SOURCE:-/home/ctj/cbf_ttt_event_curriculum_v1}
PYTHON=${PYTHON:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
MODEL=${MODEL:-/home/ctj/models/Qwen3-4B}
mkdir "$ROOT"
git rev-parse HEAD > "$ROOT/code_commit.txt"
date -Is > "$ROOT/started_at.txt"
trap 'code=$?; if (( code != 0 )); then echo "$code" > "$ROOT/failed_exit_code.txt"; date -Is > "$ROOT/failed_at.txt"; fi' EXIT
"$PYTHON" -m unittest tests.test_cbf_event_writer tests.test_cbf_event_curriculum > "$ROOT/tests.log" 2>&1
"$PYTHON" -m tasks.pack_cbf_event_warmup --source "$SOURCE" --output "$ROOT/data" --tokenizer "$MODEL" > "$ROOT/packing.log" 2>&1
(
  CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m tasks.train_cbf_event_writer train --root "$ROOT" --model "$MODEL" > "$ROOT/train.log" 2>&1 || { echo failed > "$ROOT/train_failed.txt"; exit 1; }
) &
train_pid=$!
CUDA_VISIBLE_DEVICES=1 "$PYTHON" -m tasks.train_cbf_event_writer evaluate --root "$ROOT" --model "$MODEL" > "$ROOT/evaluate.log" 2>&1 &
eval_pid=$!
status=0
wait "$train_pid" || status=1
wait "$eval_pid" || status=1
if (( status != 0 )); then exit 1; fi
"$PYTHON" -m tasks.train_cbf_event_writer summarize --root "$ROOT" > "$ROOT/summary.log" 2>&1
printf '%s\n' completed_writer_warmup > "$ROOT/terminal_status.txt"
date -Is > "$ROOT/completed_at.txt"
