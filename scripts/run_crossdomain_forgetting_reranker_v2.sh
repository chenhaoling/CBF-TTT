#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/cbf_ttt_crossdomain_forgetting_reranker_v2}
ZSRE_ROOT=${ZSRE_ROOT:-/home/ctj/cbf_ttt_zsre_counterfactual_forgetting_v1_data}
COUNTERFACT_ROOT=${COUNTERFACT_ROOT:-/home/ctj/cbf_ttt_counterfact_zero_shot_v1/data}
PREVIOUS_ROOT=${PREVIOUS_ROOT:-/home/ctj/cbf_ttt_counterfact_zero_shot_v1/evaluation}
V1_ROOT=${V1_ROOT:-/home/ctj/cbf_ttt_crossdomain_forgetting_reranker_v1/development}
CONTEXT_SOURCE=${CONTEXT_SOURCE:-/home/ctj/cbf_ttt_zsre_contextual_readout_v1/formal}
SCOPE_ROOT=${SCOPE_ROOT:-/home/ctj/cbf_ttt_zsre_decomposed_scope_gate_v1/formal}
MODEL=${MODEL:-/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt}
TOKENIZER=${TOKENIZER:-/home/ctj/models/Qwen3-4B}
PYTHON=${PYTHON:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}

if [[ -e "$ROOT" ]]; then echo "ROOT already exists: $ROOT" >&2; exit 2; fi
mkdir -p "$ROOT"; git rev-parse HEAD > "$ROOT/code_commit.txt"; date -Is > "$ROOT/started_at.txt"
trap 'code=$?; if (( code != 0 )); then echo "$code" > "$ROOT/failed_exit_code.txt"; date -Is > "$ROOT/failed_at.txt"; fi' EXIT

"$PYTHON" -m unittest tests.test_crossdomain_forgetting_reranker \
  tests.test_crossdomain_forgetting_reranker_v2 > "$ROOT/tests.log" 2>&1

CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m tasks.train_crossdomain_forgetting_reranker_v2 \
  --zsre-root "$ZSRE_ROOT" --counterfact-root "$COUNTERFACT_ROOT" \
  --previous-root "$PREVIOUS_ROOT" --v1-root "$V1_ROOT" \
  --source-root "$CONTEXT_SOURCE" --scope-root "$SCOPE_ROOT" \
  --output "$ROOT/development" --model "$MODEL" --tokenizer "$TOKENIZER" \
  --batch-size 128 > "$ROOT/development.log" 2>&1

echo completed_crossdomain_forgetting_reranker_v2 > "$ROOT/terminal_status.txt"
date -Is > "$ROOT/completed_at.txt"
