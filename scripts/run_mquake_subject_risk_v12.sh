#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/cbf_ttt_mquake_subject_risk_v12}
REPO=${REPO:-/home/ctj/cbf_ttt_joint_exp_20260927}
PYTHON=${PYTHON:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
DATA=${DATA:-/home/ctj/cbf_ttt_mquake_temporal_sessions_v1}
SOURCE=${SOURCE:-/home/ctj/cbf_ttt_zsre_contextual_readout_v1/formal}
MODEL=${MODEL:-/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt}
TOKENIZER=${TOKENIZER:-/home/ctj/models/Qwen3-4B}

cd "$REPO"
test ! -e "$ROOT"
mkdir -p "$ROOT"
"$PYTHON" -m unittest tests.test_mquake_subject_risk_v12
date --iso-8601=seconds > "$ROOT/started_at.txt"
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m tasks.train_mquake_subject_risk_v12 \
  --data-root "$DATA" --source-root "$SOURCE" --output "$ROOT/result" \
  --model "$MODEL" --tokenizer "$TOKENIZER" --batch-size 128 \
  > "$ROOT/result.log" 2>&1
"$PYTHON" -m scripts.audit_mquake_subject_risk_v12 \
  --root "$ROOT/result" --data-root "$DATA" --source-root "$SOURCE" \
  --output "$ROOT/audit.json" > "$ROOT/audit.log" 2>&1
echo completed_mquake_subject_risk_v12 > "$ROOT/terminal_status.txt"
date --iso-8601=seconds > "$ROOT/completed_at.txt"
cat "$ROOT/result/summary.json"
