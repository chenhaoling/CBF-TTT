#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT=${SOURCE_ROOT:-/home/ctj/cbf_ttt_qwen_causal_ledger_pilot_20261015}
ROOT=${ROOT:-/home/ctj/cbf_ttt_qwen_causal_ledger_context_scale_20261015}
PYTHON_BIN=${PYTHON_BIN:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
MODEL=${MODEL:-/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt}
TOKENIZER=${TOKENIZER:-/home/ctj/models/Qwen3-4B}

if [[ -e "$ROOT" ]]; then
  echo "Refusing to overwrite $ROOT" >&2
  exit 2
fi
mkdir -p "$ROOT"
git rev-parse HEAD > "$ROOT/code_commit.txt"
date --iso-8601=seconds > "$ROOT/started_at.txt"
status=failed_qwen_causal_ledger_context_scale
code=1
finish() {
  echo "$status" > "$ROOT/terminal_status.txt"
  echo "$code" > "$ROOT/runner_exit_code.txt"
  date --iso-8601=seconds > "$ROOT/completed_at.txt"
}
trap finish EXIT

declare -a pids
for shard in 0 1; do
  CUDA_VISIBLE_DEVICES=$shard "$PYTHON_BIN" -m tasks.diagnose_qwen_causal_ledger_context_scale collect \
    --root "$SOURCE_ROOT" --model "$MODEL" --tokenizer "$TOKENIZER" \
    --shard "$shard" --num-shards 2 --output "$ROOT/scale_shard${shard}.jsonl" \
    --resource-output "$ROOT/scale_resource_shard${shard}.json" \
    > "$ROOT/shard${shard}.log" 2>&1 &
  pids[$shard]=$!
done
wait "${pids[0]}"
wait "${pids[1]}"
"$PYTHON_BIN" -m tasks.diagnose_qwen_causal_ledger_context_scale summarize \
  --root "$ROOT" --output "$ROOT/summary.json" > "$ROOT/summarize.log" 2>&1
status=completed_qwen_causal_ledger_context_scale
code=0
