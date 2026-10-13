#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/cbf_ttt_mquake_temporal_transfer_v7}
REPO=${REPO:-/home/ctj/cbf_ttt_joint_exp_20260927}
PYTHON=${PYTHON:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
RAW=${RAW:-/home/ctj/data/mquake_official/MQuAKE-T.json}
DATA=${DATA:-/home/ctj/cbf_ttt_mquake_temporal_sessions_v1}
SOURCE=${SOURCE:-/home/ctj/cbf_ttt_zsre_contextual_readout_v1/formal}
V3=${V3:-/home/ctj/cbf_ttt_factorized_scope_candidate_v3/development}
V5=${V5:-/home/ctj/cbf_ttt_crossfit_calibrated_forgetting_v5/result}
MODEL=${MODEL:-/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt}
TOKENIZER=${TOKENIZER:-/home/ctj/models/Qwen3-4B}

cd "$REPO"
test ! -e "$ROOT"
mkdir -p "$ROOT"
"$PYTHON" -m unittest tests.test_build_mquake_temporal_sessions \
  tests.test_mquake_temporal_transfer_v7
date --iso-8601=seconds > "$ROOT/started_at.txt"
if test -e "$DATA"; then
  echo "reuse_existing_hashed_dataset=$DATA" > "$ROOT/build.log"
else
  "$PYTHON" -m tasks.build_mquake_temporal_sessions \
    --source "$RAW" --output "$DATA" > "$ROOT/build.log" 2>&1
fi
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m tasks.evaluate_mquake_temporal_transfer_v7 \
  --data-root "$DATA" --source-root "$SOURCE" --v3-root "$V3" --v5-root "$V5" \
  --output "$ROOT/result" --model "$MODEL" --tokenizer "$TOKENIZER" \
  --batch-size 128 > "$ROOT/result.log" 2>&1
"$PYTHON" -m scripts.audit_mquake_temporal_transfer_v7 \
  --root "$ROOT/result" --data-root "$DATA" --source-root "$SOURCE" \
  --v3-root "$V3" --v5-root "$V5" --raw-source "$RAW" \
  --output "$ROOT/audit.json" > "$ROOT/audit.log" 2>&1
echo completed_mquake_temporal_transfer_v7 > "$ROOT/terminal_status.txt"
date --iso-8601=seconds > "$ROOT/completed_at.txt"
cat "$ROOT/result/summary.json"
