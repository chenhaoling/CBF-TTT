#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/home/ctj/cbf_ttt_optimizer_audit_20261008}
REFERENCE=${REFERENCE:-/home/ctj/cbf_ttt_writer_resource_20261008}
PYTHON=${PYTHON:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
MODEL=${MODEL:-/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt}
TRAINING_DATA=${TRAINING_DATA:-/home/ctj/data/cbf_ttt_1b/mixed_1b.jsonl}
TOKENIZER=${TOKENIZER:-/home/ctj/models/Qwen3-4B}
mkdir "$ROOT"
git rev-parse HEAD > "$ROOT/code_commit.txt"
date -Is > "$ROOT/started_at.txt"
"$PYTHON" -m unittest tests.test_ttt_optimizer tests.test_ttt_train_infer > "$ROOT/tests.log" 2>&1
pids=()
for gpu in 0 1; do
  length=$((6144 * (gpu + 1)))
  CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" -m scripts.diagnose_ttt_optimizer \
    --model "$MODEL" --training-data "$TRAINING_DATA" --tokenizer "$TOKENIZER" \
    --length "$length" --reference "$REFERENCE/probe_${length}.json" \
    --output "$ROOT/audit_${length}.json" > "$ROOT/audit_${length}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if (( status != 0 )); then exit 1; fi
"$PYTHON" -m scripts.summarize_ttt_optimizer \
  --inputs "$ROOT/audit_6144.json" "$ROOT/audit_12288.json" \
  --output "$ROOT/summary.json" > "$ROOT/summary.log" 2>&1
date -Is > "$ROOT/completed_at.txt"
