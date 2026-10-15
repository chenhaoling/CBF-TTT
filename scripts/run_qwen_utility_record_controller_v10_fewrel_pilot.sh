#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/cbf_ttt_utility_record_controller_v10_fewrel_pilot_20261016}
PYTHON_BIN=${PYTHON_BIN:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
DATA_RUN=${DATA_RUN:-/home/ctj/cbf_ttt_utility_record_controller_v10_fewrel_corpus_20261016_retry1}
DATA_ROOT=${DATA_ROOT:-$DATA_RUN/data}
DATA_AUDIT=${DATA_AUDIT:-$DATA_RUN/data_audit.json}
MODEL=${MODEL:-/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt}
TOKENIZER=${TOKENIZER:-/home/ctj/models/Qwen3-4B}

if [[ -e "$ROOT" ]]; then
  echo "Refusing to overwrite $ROOT" >&2
  exit 2
fi
mkdir -p "$ROOT"
git rev-parse HEAD > "$ROOT/code_commit.txt"
date --iso-8601=seconds > "$ROOT/started_at.txt"
status=failed_qwen_utility_record_controller_v10_fewrel_pilot
code=1
finish() {
  echo "$status" > "$ROOT/terminal_status.txt"
  echo "$code" > "$ROOT/runner_exit_code.txt"
  date --iso-8601=seconds > "$ROOT/completed_at.txt"
}
trap finish EXIT

"$PYTHON_BIN" -m unittest \
  tests.test_pilot_qwen_utility_record_controller_v10_fewrel \
  > "$ROOT/tests.log" 2>&1

for shard in 0 1; do
  CUDA_VISIBLE_DEVICES=$shard "$PYTHON_BIN" \
    -m tasks.pilot_qwen_utility_record_controller_v10_fewrel collect \
    --data-root "$DATA_ROOT" --data-audit "$DATA_AUDIT" \
    --model "$MODEL" --tokenizer "$TOKENIZER" \
    --shard "$shard" --num-shards 2 \
    --output "$ROOT/labels_shard${shard}.jsonl" \
    --resource-output "$ROOT/resource_shard${shard}.json" \
    > "$ROOT/shard${shard}.log" 2>&1 &
done
wait

"$PYTHON_BIN" -m tasks.pilot_qwen_utility_record_controller_v10_fewrel summarize \
  --root "$ROOT" --data-root "$DATA_ROOT" --num-shards 2 \
  --output "$ROOT/summary.json" > "$ROOT/summary.log" 2>&1

"$PYTHON_BIN" -m scripts.audit_qwen_utility_record_controller_v10_fewrel_pilot \
  --root "$ROOT" --data-root "$DATA_ROOT" --data-audit "$DATA_AUDIT" \
  --summary "$ROOT/summary.json" --num-shards 2 \
  --output "$ROOT/pilot_audit.json" > "$ROOT/audit.log" 2>&1

status=completed_qwen_utility_record_controller_v10_fewrel_pilot
code=0
trap - EXIT
finish
