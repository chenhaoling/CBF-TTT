#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/home/ctj/cbf_ttt_entity_fusion_scope_v10}
COUNTERFACT_SOURCE=${COUNTERFACT_SOURCE:-/home/ctj/data/cbf_ttt_eval/counterfact/counterfact.json}
CORPUS_ROOT=${CORPUS_ROOT:-/home/ctj/cbf_ttt_relation_supervised_scope_v5/corpus}
V4_ROOT=${V4_ROOT:-/home/ctj/cbf_ttt_scoped_delta_ranker_v4/development}
V5_ROOT=${V5_ROOT:-/home/ctj/cbf_ttt_relation_supervised_scope_v5/development}
V8_ROOT=${V8_ROOT:-/home/ctj/cbf_ttt_text_only_scope_calibrator_v8/development}
V9_ROOT=${V9_ROOT:-/home/ctj/cbf_ttt_learned_entity_scope_v9/development}
CONTEXT_SOURCE=${CONTEXT_SOURCE:-/home/ctj/cbf_ttt_zsre_contextual_readout_v1/formal}
MODEL=${MODEL:-/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt}
TOKENIZER=${TOKENIZER:-/home/ctj/models/Qwen3-4B}
PYTHON=${PYTHON:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
if [[ -e "$ROOT" ]];then echo "ROOT already exists: $ROOT" >&2;exit 2;fi
mkdir -p "$ROOT";git rev-parse HEAD > "$ROOT/code_commit.txt";date -Is > "$ROOT/started_at.txt"
trap 'code=$?;if (( code != 0 ));then echo "$code" > "$ROOT/failed_exit_code.txt";date -Is > "$ROOT/failed_at.txt";fi' EXIT
"$PYTHON" -m unittest tests.test_entity_fusion_scope_v10 > "$ROOT/tests.log" 2>&1
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m tasks.train_entity_fusion_scope_v10 \
 --corpus-root "$CORPUS_ROOT" --counterfact-source "$COUNTERFACT_SOURCE" \
 --v4-root "$V4_ROOT" --v5-root "$V5_ROOT" --v8-root "$V8_ROOT" --v9-root "$V9_ROOT" \
 --source-root "$CONTEXT_SOURCE" --output "$ROOT/development" --model "$MODEL" --tokenizer "$TOKENIZER" \
 --batch-size 256 --pair-batch-size 128 > "$ROOT/development.log" 2>&1
echo completed_entity_fusion_scope_v10 > "$ROOT/terminal_status.txt";date -Is > "$ROOT/completed_at.txt"

