#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/cbf_ttt_relation_supervised_scope_v5}
ZSRE_ROOT=${ZSRE_ROOT:-/home/ctj/cbf_ttt_zsre_counterfactual_forgetting_v1_data}
COUNTERFACT_ROOT=${COUNTERFACT_ROOT:-/home/ctj/cbf_ttt_counterfact_zero_shot_v1/data}
COUNTERFACT_SOURCE=${COUNTERFACT_SOURCE:-/home/ctj/data/cbf_ttt_eval/counterfact/counterfact.json}
V4_ROOT=${V4_ROOT:-/home/ctj/cbf_ttt_scoped_delta_ranker_v4/development}
CONTEXT_SOURCE=${CONTEXT_SOURCE:-/home/ctj/cbf_ttt_zsre_contextual_readout_v1/formal}
MODEL=${MODEL:-/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt}
TOKENIZER=${TOKENIZER:-/home/ctj/models/Qwen3-4B}
PYTHON=${PYTHON:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}

if [[ -e "$ROOT" ]]; then echo "ROOT already exists: $ROOT" >&2; exit 2; fi
mkdir -p "$ROOT"; git rev-parse HEAD > "$ROOT/code_commit.txt"; date -Is > "$ROOT/started_at.txt"
trap 'code=$?; if (( code != 0 )); then echo "$code" > "$ROOT/failed_exit_code.txt"; date -Is > "$ROOT/failed_at.txt"; fi' EXIT

"$PYTHON" -m unittest tests.test_scoped_delta_corpus \
  tests.test_relation_supervised_scope_v5 > "$ROOT/tests.log" 2>&1

"$PYTHON" -m tasks.build_scoped_delta_corpus \
  --zsre-root "$ZSRE_ROOT" --counterfact-root "$COUNTERFACT_ROOT" \
  --output "$ROOT/corpus" > "$ROOT/corpus.log" 2>&1

CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m tasks.train_relation_supervised_scope_v5 \
  --corpus-root "$ROOT/corpus" --counterfact-source "$COUNTERFACT_SOURCE" \
  --v4-root "$V4_ROOT" --source-root "$CONTEXT_SOURCE" \
  --output "$ROOT/development" --model "$MODEL" --tokenizer "$TOKENIZER" \
  --batch-size 256 > "$ROOT/development.log" 2>&1

echo completed_relation_supervised_scope_v5 > "$ROOT/terminal_status.txt"
date -Is > "$ROOT/completed_at.txt"

