#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/home/ctj/cbf_ttt_token_span_relation_scope_v11}
COUNTERFACT_SOURCE=${COUNTERFACT_SOURCE:-/home/ctj/data/cbf_ttt_eval/counterfact/counterfact.json}
CORPUS_ROOT=${CORPUS_ROOT:-/home/ctj/cbf_ttt_relation_supervised_scope_v5/corpus}
V5_ROOT=${V5_ROOT:-/home/ctj/cbf_ttt_relation_supervised_scope_v5/development}
V10_ROOT=${V10_ROOT:-/home/ctj/cbf_ttt_entity_fusion_scope_v10/development}
CONTEXT_SOURCE=${CONTEXT_SOURCE:-/home/ctj/cbf_ttt_zsre_contextual_readout_v1/formal}
MODEL=${MODEL:-/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt}
TOKENIZER=${TOKENIZER:-/home/ctj/models/Qwen3-4B}
PYTHON=${PYTHON:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
if [[ -e "$ROOT" ]]; then echo "ROOT already exists: $ROOT" >&2; exit 2; fi
mkdir -p "$ROOT"
git rev-parse HEAD > "$ROOT/code_commit.txt"
date -Is > "$ROOT/started_at.txt"
trap 'code=$?; if (( code != 0 )); then echo "$code" > "$ROOT/failed_exit_code.txt"; date -Is > "$ROOT/failed_at.txt"; fi' EXIT
"$PYTHON" -m unittest tests.test_token_span_relation_scope_v11 > "$ROOT/tests.log" 2>&1
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m tasks.train_token_span_relation_scope_v11 \
  --corpus-root "$CORPUS_ROOT" --counterfact-source "$COUNTERFACT_SOURCE" \
  --v5-root "$V5_ROOT" --v10-root "$V10_ROOT" --source-root "$CONTEXT_SOURCE" \
  --output "$ROOT/development" --model "$MODEL" --tokenizer "$TOKENIZER" \
  --extract-batch-size 32 --relation-batch-size 256 --token-batch-size 65536 \
  --max-length 128 > "$ROOT/development.log" 2>&1
echo completed_token_span_relation_scope_v11 > "$ROOT/terminal_status.txt"
date -Is > "$ROOT/completed_at.txt"
