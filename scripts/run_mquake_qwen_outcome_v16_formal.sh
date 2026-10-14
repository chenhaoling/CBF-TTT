#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/cbf_ttt_mquake_qwen_outcome_v16_formal}
REPO=${REPO:-/home/ctj/cbf_ttt_joint_exp_20260927}
PYTHON=${PYTHON:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
DATA=${DATA:-/home/ctj/cbf_ttt_mquake_temporal_sessions_v1}
SAMPLE=${SAMPLE:-/home/ctj/cbf_ttt_mquake_qwen_outcome_v16}
STATES=${STATES:-/home/ctj/cbf_ttt_mquake_qwen_outcome_v16_formal_states/result}
MODEL=${MODEL:-/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt}
TOKENIZER=${TOKENIZER:-/home/ctj/models/Qwen3-4B}

cd "$REPO"
test ! -e "$ROOT"
mkdir -p "$ROOT"
trap 'code=$?; echo "$code" > "$ROOT/runner_exit_code.txt"' EXIT
date --iso-8601=seconds > "$ROOT/started_at.txt"
export PYTHONDONTWRITEBYTECODE=1
"$PYTHON" -m unittest \
  tests.test_evaluate_mquake_qwen_outcome_v16 \
  tests.test_evaluate_mquake_qwen_outcome_v16_formal \
  tests.test_audit_mquake_qwen_outcome_v16_formal
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m scripts.check_mquake_qwen_outcome_v16_generation \
  --model "$MODEL" --tokenizer "$TOKENIZER" --output "$ROOT/generation_bridge.json" \
  > "$ROOT/generation_bridge.log" 2>&1

CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m tasks.evaluate_mquake_qwen_outcome_v16_formal collect \
  --sample-root "$SAMPLE" --state-root "$STATES" --data-root "$DATA" \
  --model "$MODEL" --tokenizer "$TOKENIZER" --shard 0 --num-shards 2 \
  --output "$ROOT/shard0.jsonl" > "$ROOT/shard0.log" 2>&1 &
pid0=$!
CUDA_VISIBLE_DEVICES=1 "$PYTHON" -m tasks.evaluate_mquake_qwen_outcome_v16_formal collect \
  --sample-root "$SAMPLE" --state-root "$STATES" --data-root "$DATA" \
  --model "$MODEL" --tokenizer "$TOKENIZER" --shard 1 --num-shards 2 \
  --output "$ROOT/shard1.jsonl" > "$ROOT/shard1.log" 2>&1 &
pid1=$!
status=0
wait "$pid0" || status=$?
wait "$pid1" || status=$?
if [[ "$status" -ne 0 ]]; then
  exit "$status"
fi

"$PYTHON" -m tasks.evaluate_mquake_qwen_outcome_v16_formal summarize \
  --sample-root "$SAMPLE" --state-root "$STATES" \
  --inputs "$ROOT/shard0.jsonl" "$ROOT/shard1.jsonl" \
  --generation-bridge "$ROOT/generation_bridge.json" \
  --output "$ROOT/summary.json" > "$ROOT/summary.log" 2>&1
"$PYTHON" -m scripts.audit_mquake_qwen_outcome_v16_formal \
  --data-root "$DATA" --sample-root "$SAMPLE" --state-root "$STATES" \
  --model "$MODEL" --tokenizer "$TOKENIZER" \
  --inputs "$ROOT/shard0.jsonl" "$ROOT/shard1.jsonl" \
  --summary-path "$ROOT/summary.json" --output "$ROOT/audit.json" \
  --generation-bridge "$ROOT/generation_bridge.json" \
  > "$ROOT/audit.log" 2>&1
echo completed_mquake_qwen_outcome_v16_formal > "$ROOT/terminal_status.txt"
date --iso-8601=seconds > "$ROOT/completed_at.txt"
cat "$ROOT/summary.json"
