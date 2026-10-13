#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/cbf_ttt_longmemeval_forgetting_transfer_v1}
REPO=${REPO:-/home/ctj/cbf_ttt_joint_exp_20260927}
PYTHON=${PYTHON:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
DATA=${DATA:-/home/ctj/data/cbf_ttt_eval/longmemeval_s/longmemeval_s_cleaned.json}
SOURCE=${SOURCE:-/home/ctj/cbf_ttt_zsre_contextual_readout_v1/formal}
CONTROLLER=${CONTROLLER:-/home/ctj/cbf_ttt_zsre_counterfactual_forgetting_controller_loso_v3/formal}
FINAL=${FINAL:-/home/ctj/cbf_ttt_zsre_forgetting_final_test_v1/test/summary.json}
MODEL=${MODEL:-/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt}
TOKENIZER=${TOKENIZER:-/home/ctj/models/Qwen3-4B}

cd "$REPO"
test ! -e "$ROOT"
mkdir -p "$ROOT"
"$PYTHON" -m unittest tests.test_longmemeval_forgetting_transfer
date --iso-8601=seconds > "$ROOT/started_at.txt"

pids=()
for shard in 0 1; do
  CUDA_VISIBLE_DEVICES=$shard "$PYTHON" -m tasks.evaluate_longmemeval_forgetting_transfer collect \
    --data "$DATA" --source-root "$SOURCE" --controller-root "$CONTROLLER" \
    --final-summary "$FINAL" --model "$MODEL" --tokenizer "$TOKENIZER" \
    --shard "$shard" --num-shards 2 --batch-size 4 \
    --output "$ROOT/shard_${shard}.jsonl" > "$ROOT/shard_${shard}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

"$PYTHON" -m tasks.evaluate_longmemeval_forgetting_transfer summarize \
  --data "$DATA" --root "$ROOT" --num-shards 2 > "$ROOT/summarize.log" 2>&1
echo completed_longmemeval_forgetting_transfer_v1 > "$ROOT/terminal_status.txt"
date --iso-8601=seconds > "$ROOT/completed_at.txt"
cat "$ROOT/summary.json"
