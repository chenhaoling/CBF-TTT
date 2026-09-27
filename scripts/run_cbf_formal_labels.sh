#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "Usage: bash scripts/run_cbf_formal_labels.sh MODEL SCENARIO_DIR SHARD GPU [forget|write]" >&2
  exit 2
fi

model=$1
scenario_dir=$2
shard=$3
gpu=$4
update_rule=${5:-forget}
if [[ $update_rule != forget && $update_rule != write ]]; then
  echo "Update rule must be forget or write" >&2
  exit 2
fi
if [[ ! $shard =~ ^[0-9]+$ || ! $gpu =~ ^[0-9]+$ ]]; then
  echo "SHARD and GPU must be nonnegative integers" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES=$gpu
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
for split in train dev test; do
  data="$scenario_dir/shards/${split}_shard${shard}.jsonl"
  output="$scenario_dir/${split}_shard${shard}_labels.jsonl"
  if [[ ! -s $data ]]; then
    echo "Missing or empty scenario shard: $data" >&2
    exit 1
  fi
  if [[ -e $output || -e $output.summary.json ]]; then
    echo "Output exists; refusing to overwrite: $output" >&2
    exit 1
  fi
  echo "Collecting split=$split shard=$shard gpu=$gpu at $(date -Is)"
  python -m tasks.cbf_ttt collect \
    --model "$model" --dtype bfloat16 --data "$data" --split "$split" \
    --output "$output" --grid 0,0.5,1 --every 2 --update-rule "$update_rule"
done
echo "Finished shard=$shard gpu=$gpu at $(date -Is)"
