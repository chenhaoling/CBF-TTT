#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: bash scripts/run_cbf_controller_learning_curve.sh LABEL_DIR STUDY_DIR" >&2
  exit 2
fi

label_dir=$1
study_dir=$2
mkdir -p "$study_dir"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

for groups in 250 500 1000; do
  train="$study_dir/train_groups_${groups}.jsonl"
  if [[ ! -s $train ]]; then
    echo "Missing training subset: $train" >&2
    exit 1
  fi
  for seed in 42 43 44; do
    output="$study_dir/controller_g${groups}_s${seed}.pt"
    if [[ -e $output || -e $output.metrics.json ]]; then
      echo "Output exists; refusing to overwrite: $output" >&2
      exit 1
    fi
    echo "Training groups=$groups seed=$seed at $(date -Is)"
    python -m tasks.cbf_ttt train \
      --train-samples "$train" \
      --dev-samples "$label_dir/dev_shard0_labels.jsonl" "$label_dir/dev_shard1_labels.jsonl" \
      --layers 0 6 12 18 24 30 35 \
      --output "$output" --epochs 20 --batch-size 64 --lr 0.001 \
      --width 128 --semantic-size 64 --seed "$seed"
  done
done
echo "Controller learning curve training finished at $(date -Is)"
