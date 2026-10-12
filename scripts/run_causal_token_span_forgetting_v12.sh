#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/home/ctj/cbf_ttt_causal_token_span_forgetting_v12}
COUNTERFACT_ROOT=${COUNTERFACT_ROOT:-/home/ctj/cbf_ttt_counterfact_zero_shot_v1/data}
V5_ROOT=${V5_ROOT:-/home/ctj/cbf_ttt_relation_supervised_scope_v5/development}
V11_ROOT=${V11_ROOT:-/home/ctj/cbf_ttt_token_span_relation_scope_v11/development}
PREVIOUS_ROOT=${PREVIOUS_ROOT:-/home/ctj/cbf_ttt_counterfact_zero_shot_v1/evaluation}
CONTEXT_SOURCE=${CONTEXT_SOURCE:-/home/ctj/cbf_ttt_zsre_contextual_readout_v1/formal}
SCOPE_ROOT=${SCOPE_ROOT:-/home/ctj/cbf_ttt_zsre_decomposed_scope_gate_v1/formal}
MODEL=${MODEL:-/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt}
TOKENIZER=${TOKENIZER:-/home/ctj/models/Qwen3-4B}
PYTHON=${PYTHON:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
if [[ -e "$ROOT" ]]; then echo "ROOT already exists: $ROOT" >&2; exit 2; fi
mkdir -p "$ROOT"
git rev-parse HEAD > "$ROOT/code_commit.txt"
date -Is > "$ROOT/started_at.txt"
trap 'code=$?; if (( code != 0 )); then echo "$code" > "$ROOT/failed_exit_code.txt"; date -Is > "$ROOT/failed_at.txt"; fi' EXIT
"$PYTHON" -m unittest tests.test_causal_token_span_forgetting_v12 \
  tests.test_token_span_relation_scope_v11 > "$ROOT/tests.log" 2>&1
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m tasks.train_causal_token_span_forgetting_v12 \
  --counterfact-root "$COUNTERFACT_ROOT" --v5-root "$V5_ROOT" --v11-root "$V11_ROOT" \
  --previous-root "$PREVIOUS_ROOT" --source-root "$CONTEXT_SOURCE" --scope-root "$SCOPE_ROOT" \
  --output "$ROOT/development" --model "$MODEL" --tokenizer "$TOKENIZER" \
  --extract-batch-size 32 --readout-batch-size 256 --pair-batch-size 128 \
  > "$ROOT/development.log" 2>&1
echo completed_causal_token_span_forgetting_v12 > "$ROOT/terminal_status.txt"
date -Is > "$ROOT/completed_at.txt"
