#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/home/ctj/cbf_ttt_qwen_lowrank_forgetting_v1}
MODEL=${MODEL:-/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt}
TOKENIZER=${TOKENIZER:-/home/ctj/models/Qwen3-4B}
PYTHON=${PYTHON:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
if [[ -e "$ROOT" ]]; then echo "ROOT already exists: $ROOT" >&2; exit 2; fi
mkdir -p "$ROOT"; git rev-parse HEAD > "$ROOT/code_commit.txt"; date -Is > "$ROOT/started_at.txt"
trap 'code=$?; if (( code != 0 )); then echo "$code" > "$ROOT/failed_exit_code.txt"; date -Is > "$ROOT/failed_at.txt"; fi' EXIT
"$PYTHON" -m unittest tests.test_cbf_qwen_lowrank_forgetting tests.test_cbf_lowrank_forgetting tests.test_cbf_structured_forgetting tests.test_cbf_event_curriculum
"$PYTHON" -m tasks.build_cbf_event_curriculum --output "$ROOT/data" --seed 20261101 --train-groups 128 --dev-groups 32 --test-groups 32
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m tasks.train_cbf_qwen_lowrank_forgetting \
  --data-root "$ROOT/data" --output "$ROOT/formal" --model "$MODEL" --tokenizer "$TOKENIZER" \
  --dim 128 --epochs 300 --seed 2321 > "$ROOT/formal.log" 2>&1
"$PYTHON" -m scripts.audit_cbf_qwen_lowrank_forgetting --root "$ROOT/formal" --data-root "$ROOT/data" --output "$ROOT/execution_audit.json"
echo completed_qwen_lowrank_forgetting_v1 > "$ROOT/terminal_status.txt"; date -Is > "$ROOT/completed_at.txt"
