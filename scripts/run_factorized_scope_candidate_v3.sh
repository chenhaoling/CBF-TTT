#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/cbf_ttt_factorized_scope_candidate_v3}
REPO=${REPO:-/home/ctj/cbf_ttt_joint_exp_20260927}
PYTHON=${PYTHON:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
DATA=${DATA:-/home/ctj/cbf_ttt_long_session_sparse_corpus_v1}
SOURCE=${SOURCE:-/home/ctj/cbf_ttt_zsre_contextual_readout_v1/formal}
V2=${V2:-/home/ctj/cbf_ttt_long_session_candidate_v2/development}
MODEL=${MODEL:-/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt}
TOKENIZER=${TOKENIZER:-/home/ctj/models/Qwen3-4B}

cd "$REPO"
test ! -e "$ROOT"
mkdir -p "$ROOT"
"$PYTHON" -m unittest tests.test_train_factorized_scope_candidate_v3
date --iso-8601=seconds > "$ROOT/started_at.txt"
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m tasks.train_factorized_scope_candidate_v3 \
  --data-root "$DATA" --source-root "$SOURCE" --v2-root "$V2" \
  --output "$ROOT/development" --model "$MODEL" --tokenizer "$TOKENIZER" \
  --batch-size 128 > "$ROOT/development.log" 2>&1
"$PYTHON" -m scripts.audit_factorized_scope_candidate_v3 \
  --root "$ROOT/development" --data-root "$DATA" --source-root "$SOURCE" \
  --v2-root "$V2" --output "$ROOT/audit.json" > "$ROOT/audit.log" 2>&1
echo completed_factorized_scope_candidate_v3 > "$ROOT/terminal_status.txt"
date --iso-8601=seconds > "$ROOT/completed_at.txt"
cat "$ROOT/development/summary.json"
