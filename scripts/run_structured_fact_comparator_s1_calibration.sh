#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
REPO=${REPO:-/home/ctj/cbf_ttt_joint_exp_20260927}
DATA=${DATA:-/home/ctj/cbf_ttt_structured_fact_comparator_s1_data_20261015}
TRAIN=${TRAIN:-/home/ctj/cbf_ttt_structured_fact_comparator_s1_train_20261015}
MODEL=${MODEL:-/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt}
TOKENIZER=${TOKENIZER:-/home/ctj/models/Qwen3-4B}
OUTPUT=${OUTPUT:-/home/ctj/cbf_ttt_structured_fact_comparator_s1_calibration_20261015}

cd "$REPO"
"$PY" -m unittest \
  tests.test_score_structured_fact_comparator_s1 \
  tests.test_select_structured_fact_comparator_s1 \
  tests.test_audit_structured_fact_comparator_s1_training \
  tests.test_audit_structured_fact_comparator_s1_selection -v
"$PY" scripts/audit_structured_fact_comparator_s1_training.py \
  --root "$TRAIN" --data-root "$DATA" --output "$TRAIN/training_audit.json"
test ! -e "$OUTPUT"
mkdir -p "$OUTPUT"
date -Iseconds > "$OUTPUT/started_at.txt"

run_arm() {
  local name=$1
  local adapter=${2:-}
  local directory="$OUTPUT/$name"
  mkdir -p "$directory"
  local extra=()
  if [[ -n "$adapter" ]]; then
    extra=(--adapter "$adapter")
  fi
  CUDA_VISIBLE_DEVICES=0 "$PY" -m tasks.score_structured_fact_comparator_s1 \
    --data-root "$DATA" --model "$MODEL" --tokenizer "$TOKENIZER" \
    --split calibration --mode candidate --shard 0 --num-shards 2 --pair-batch 4 \
    --output "$directory/shard_0.jsonl" "${extra[@]}" \
    > "$directory/shard_0.log" 2>&1 &
  local pid0=$!
  CUDA_VISIBLE_DEVICES=1 "$PY" -m tasks.score_structured_fact_comparator_s1 \
    --data-root "$DATA" --model "$MODEL" --tokenizer "$TOKENIZER" \
    --split calibration --mode candidate --shard 1 --num-shards 2 --pair-batch 4 \
    --output "$directory/shard_1.jsonl" "${extra[@]}" \
    > "$directory/shard_1.log" 2>&1 &
  local pid1=$!
  wait "$pid0"
  wait "$pid1"
}

run_arm zero_shot
run_arm epoch_0_25 "$TRAIN/checkpoint_epoch_0.25"
run_arm epoch_0_50 "$TRAIN/checkpoint_epoch_0.50"
run_arm epoch_1_00 "$TRAIN/checkpoint_epoch_1.00"
run_arm epoch_2_00 "$TRAIN/checkpoint_epoch_2.00"

"$PY" -m tasks.select_structured_fact_comparator_s1 \
  --score-root "$OUTPUT" --data-root "$DATA" \
  --arms zero_shot,epoch_0_25,epoch_0_50,epoch_1_00,epoch_2_00 \
  --output "$OUTPUT/selection.json" > "$OUTPUT/selection.log" 2>&1
"$PY" scripts/audit_structured_fact_comparator_s1_selection.py \
  --score-root "$OUTPUT" --data-root "$DATA" \
  --output "$OUTPUT/selection_audit.json" > "$OUTPUT/selection_audit.log" 2>&1
date -Iseconds > "$OUTPUT/completed_at.txt"
printf '%s\n' completed_calibration_before_development > "$OUTPUT/terminal_status.txt"
