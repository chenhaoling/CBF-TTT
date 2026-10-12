#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/home/ctj/cbf_ttt_zsre_contextual_readout_v1}
SOURCE=${SOURCE:-/home/ctj/data/cbf_ttt_eval/zsre/benchmark/ZsRE/ZsRE-test-all.json}
MODEL=${MODEL:-/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt}
TOKENIZER=${TOKENIZER:-/home/ctj/models/Qwen3-4B}
PYTHON=${PYTHON:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
if [[ -e "$ROOT" ]]; then echo "ROOT already exists: $ROOT" >&2; exit 2; fi
mkdir -p "$ROOT"; git rev-parse HEAD > "$ROOT/code_commit.txt"; date -Is > "$ROOT/started_at.txt"
trap 'code=$?; if (( code != 0 )); then echo "$code" > "$ROOT/failed_exit_code.txt"; date -Is > "$ROOT/failed_at.txt"; fi' EXIT
"$PYTHON" -m unittest tests.test_zsre_contextual_readout
"$PYTHON" -m tasks.build_zsre_memory_benchmark --source "$SOURCE" --output "$ROOT/data"
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m tasks.evaluate_zsre_contextual_readout \
  --data-root "$ROOT/data" --output "$ROOT/formal" --model "$MODEL" --tokenizer "$TOKENIZER" \
  --batch-size 128 > "$ROOT/formal.log" 2>&1
"$PYTHON" -m scripts.audit_zsre_contextual_readout \
  --root "$ROOT/formal" --data-root "$ROOT/data" --output "$ROOT/execution_audit.json"
echo completed_zsre_contextual_readout_v1 > "$ROOT/terminal_status.txt"; date -Is > "$ROOT/completed_at.txt"
