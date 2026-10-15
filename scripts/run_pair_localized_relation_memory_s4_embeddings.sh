#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "usage: $0 DATA_ROOT OUTPUT_ROOT MODEL TOKENIZER [DATASET_CSV]" >&2
  exit 2
fi

data_root=$1
output_root=$2
model=$3
tokenizer=$4
dataset_csv=${5:-train_candidates,calibration_candidates,train_controls,calibration_controls}
python_bin=${PYTHON_BIN:-python}
batch_size=${BATCH_SIZE:-16}
max_length=${MAX_LENGTH:-1024}

mkdir -p "$output_root"
IFS=',' read -r -a datasets <<< "$dataset_csv"
for dataset in "${datasets[@]}"; do
  log0="$output_root/${dataset}_shard_0.log"
  log1="$output_root/${dataset}_shard_1.log"
  CUDA_VISIBLE_DEVICES=0 "$python_bin" -m tasks.extract_pair_localized_relation_memory_s4_embeddings \
    --data-root "$data_root" --model "$model" --tokenizer "$tokenizer" \
    --dataset "$dataset" --shard 0 --num-shards 2 --batch-size "$batch_size" \
    --max-length "$max_length" --output "$output_root/${dataset}_shard_0.npz" \
    >"$log0" 2>&1 &
  pid0=$!
  CUDA_VISIBLE_DEVICES=1 "$python_bin" -m tasks.extract_pair_localized_relation_memory_s4_embeddings \
    --data-root "$data_root" --model "$model" --tokenizer "$tokenizer" \
    --dataset "$dataset" --shard 1 --num-shards 2 --batch-size "$batch_size" \
    --max-length "$max_length" --output "$output_root/${dataset}_shard_1.npz" \
    >"$log1" 2>&1 &
  pid1=$!
  status=0
  wait "$pid0" || status=$?
  wait "$pid1" || status=$?
  if [[ $status -ne 0 ]]; then
    echo "S4 extraction failed for $dataset; inspect $log0 and $log1" >&2
    exit "$status"
  fi
  tail -n 1 "$log0"
  tail -n 1 "$log1"
done
