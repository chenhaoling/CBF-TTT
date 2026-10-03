#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/home/ctj/cbf_ttt_paired_fact_20260930}
OUT=${OUT:-$ROOT/path_diagnostic_20261003}
MODEL=${MODEL:-/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt}
PYTHON=${PYTHON:-python}
mkdir "$OUT"
git rev-parse HEAD > "$OUT/code_commit.txt"
"$PYTHON" -m unittest tests.test_cbf_writer_path > "$OUT/tests.log" 2>&1
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m scripts.diagnose_cbf_writer_path --root "$ROOT" --model "$MODEL" --dtype bfloat16 --output "$OUT/bfloat16" > "$OUT/bfloat16.log" 2>&1 &
pid0=$!
CUDA_VISIBLE_DEVICES=1 "$PYTHON" -m scripts.diagnose_cbf_writer_path --root "$ROOT" --model "$MODEL" --dtype float32 --output "$OUT/float32" > "$OUT/float32.log" 2>&1 &
pid1=$!
failed=0
wait "$pid0" || failed=1
wait "$pid1" || failed=1
[[ "$failed" == 0 ]] || exit 1
date -u +%FT%TZ > "$OUT/completed.txt"
