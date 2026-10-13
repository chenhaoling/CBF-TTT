#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/cbf_ttt_crossfit_calibrated_forgetting_v5}
REPO=${REPO:-/home/ctj/cbf_ttt_joint_exp_20260927}
PYTHON=${PYTHON:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
DATA=${DATA:-/home/ctj/cbf_ttt_long_session_sparse_corpus_v1}
SOURCE=${SOURCE:-/home/ctj/cbf_ttt_zsre_contextual_readout_v1/formal}
V1=${V1:-/home/ctj/cbf_ttt_long_session_sparse_gate_v1/development}
V3=${V3:-/home/ctj/cbf_ttt_factorized_scope_candidate_v3/development}
MODEL=${MODEL:-/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt}
TOKENIZER=${TOKENIZER:-/home/ctj/models/Qwen3-4B}

cd "$REPO"
test ! -e "$ROOT"
mkdir -p "$ROOT"
"$PYTHON" -m unittest tests.test_train_crossfit_calibrated_forgetting_v5
date --iso-8601=seconds > "$ROOT/started_at.txt"
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m tasks.train_crossfit_calibrated_forgetting_v5 \
  --data-root "$DATA" --source-root "$SOURCE" --v1-root "$V1" --v3-root "$V3" \
  --output "$ROOT/result" --model "$MODEL" --tokenizer "$TOKENIZER" \
  --batch-size 128 > "$ROOT/result.log" 2>&1
"$PYTHON" -m scripts.audit_crossfit_calibrated_forgetting_v5 \
  --root "$ROOT/result" --data-root "$DATA" --source-root "$SOURCE" \
  --v1-root "$V1" --v3-root "$V3" --output "$ROOT/audit.json" \
  > "$ROOT/audit.log" 2>&1
echo completed_crossfit_calibrated_forgetting_v5 > "$ROOT/terminal_status.txt"
date --iso-8601=seconds > "$ROOT/completed_at.txt"
cat "$ROOT/result/summary.json"
