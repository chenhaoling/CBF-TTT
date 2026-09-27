#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: bash scripts/run_cbf_joint_pilot.sh MODEL PILOT_DIR SHARD GPU" >&2
  exit 2
fi

model=$1
pilot_dir=$2
shard=$3
gpu=$4
if [[ ! $shard =~ ^[0-9]+$ || ! $gpu =~ ^[0-9]+$ ]]; then
  echo "SHARD and GPU must be nonnegative integers" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES=$gpu
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
for split in train dev test; do
  data="$pilot_dir/shards/${split}_shard${shard}.jsonl"
  output="$pilot_dir/${split}_shard${shard}_joint_labels.jsonl"
  if [[ ! -s $data ]]; then
    echo "Missing scenario shard: $data" >&2
    exit 1
  fi
  if [[ -e $output || -e $output.summary.json ]]; then
    echo "Refusing to overwrite: $output" >&2
    exit 1
  fi
  echo "Joint labels split=$split shard=$shard gpu=$gpu at $(date -Is)"
  python -m tasks.cbf_ttt collect-joint \
    --model "$model" --dtype bfloat16 --data "$data" --split "$split" \
    --output "$output" --grid 0,0.5,1 --every 2
done
echo "Joint pilot shard=$shard complete at $(date -Is)"
