#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/home/ctj/cbf_ttt_memory_stability_20261008}
SOURCE=${SOURCE:-/home/ctj/cbf_ttt_selective_forgetting_20261008_r3}
PYTHON=${PYTHON:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
MODEL=${MODEL:-/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt}
mkdir "$ROOT"
git rev-parse HEAD > "$ROOT/code_commit.txt"
date -Is > "$ROOT/started_at.txt"
"$PYTHON" -m unittest tests.test_cbf_memory_stability tests.test_cbf_selective tests.test_cbf_ttt \
  > "$ROOT/tests.log" 2>&1
pids=()
for gpu in 0 1; do
  CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" -m scripts.diagnose_cbf_memory_stability collect \
    --model "$MODEL" --data "$SOURCE/scenes.jsonl" --output "$ROOT/rows_${gpu}.jsonl" --shard "$gpu" \
    > "$ROOT/collect_${gpu}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if (( status != 0 )); then exit 1; fi
"$PYTHON" -m scripts.diagnose_cbf_memory_stability summarize \
  --inputs "$ROOT/rows_0.jsonl" "$ROOT/rows_1.jsonl" \
  --reference "$SOURCE/labels_0.jsonl" "$SOURCE/labels_1.jsonl" \
  --output "$ROOT/summary.json" > "$ROOT/summary.log" 2>&1
date -Is > "$ROOT/completed_at.txt"
