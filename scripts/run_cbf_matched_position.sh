#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/home/ctj/cbf_ttt_matched_position_20261009}
REFERENCE=${REFERENCE:-/home/ctj/cbf_ttt_record_interference_20261009}
PYTHON=${PYTHON:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
MODEL=${MODEL:-/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt}
mkdir "$ROOT"
git rev-parse HEAD > "$ROOT/code_commit.txt"
date -Is > "$ROOT/started_at.txt"
trap 'code=$?; if (( code != 0 )); then echo "$code" > "$ROOT/failed_exit_code.txt"; date -Is > "$ROOT/failed_at.txt"; fi' EXIT
"$PYTHON" -m unittest tests.test_cbf_matched_position tests.test_cbf_record_interference tests.test_cbf_memory_value tests.test_cbf_source_exclusion > "$ROOT/tests.log" 2>&1
"$PYTHON" -m tasks.cbf_matched_position build --reference "$REFERENCE" --output "$ROOT/data" > "$ROOT/build.log" 2>&1
pids=()
for gpu in 0 1; do
  CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" -m tasks.cbf_matched_position collect --model "$MODEL" \
    --data "$ROOT/data/scenes.jsonl" --shard "$gpu" --output "$ROOT/rows_${gpu}.jsonl" > "$ROOT/collect_${gpu}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if (( status != 0 )); then exit 1; fi
"$PYTHON" -m tasks.cbf_matched_position summarize --data "$ROOT/data/scenes.jsonl" \
  --inputs "$ROOT/rows_0.jsonl" "$ROOT/rows_1.jsonl" \
  --reference "$REFERENCE/rows_0.jsonl" "$REFERENCE/rows_1.jsonl" --output "$ROOT/summary.json" > "$ROOT/summary.log" 2>&1
date -Is > "$ROOT/completed_at.txt"
