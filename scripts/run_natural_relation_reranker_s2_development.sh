#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
REPO=${REPO:-/home/ctj/cbf_ttt_joint_exp_20260927}
DATA=${DATA:-/home/ctj/cbf_ttt_natural_relation_reranker_s2_data_20261015}
MODEL=${MODEL:-/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt}
TOKENIZER=${TOKENIZER:-/home/ctj/models/Qwen3-4B}
CALIBRATION=${CALIBRATION:-/home/ctj/cbf_ttt_natural_relation_reranker_s2_calibration_20261015}
OUTPUT=${OUTPUT:-/home/ctj/cbf_ttt_natural_relation_reranker_s2_development_20261015}

cd "$REPO"
"$PY" -m unittest \
  tests.test_evaluate_natural_relation_reranker_s2_development \
  tests.test_audit_natural_relation_reranker_s2_development -v
"$PY" - "$CALIBRATION/selection.json" "$CALIBRATION/selection_audit.json" <<'PY'
import json, sys
selection, audit = map(lambda path: json.load(open(path)), sys.argv[1:])
assert audit["passed"] is True and audit["selection_sha256"]
assert selection["development_loaded"] is False and selection["semeval_test_loaded"] is False
PY
test ! -e "$OUTPUT"
mkdir -p "$OUTPUT/embeddings" "$OUTPUT/selected" "$OUTPUT/zero_shot"
date -Iseconds > "$OUTPUT/started_at.txt"
SELECTED_ADAPTER=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_adapter"])' \
  "$CALIBRATION/selection.json")

run_embeddings() {
  CUDA_VISIBLE_DEVICES=0 "$PY" -m tasks.extract_natural_relation_reranker_s2_embeddings \
    --data-root "$DATA" --model "$MODEL" --tokenizer "$TOKENIZER" \
    --split development --selection "$CALIBRATION/selection.json" \
    --shard 0 --num-shards 2 --batch-size 16 \
    --output "$OUTPUT/embeddings/shard_0.npz" \
    > "$OUTPUT/embeddings/shard_0.log" 2>&1 &
  local pid0=$!
  CUDA_VISIBLE_DEVICES=1 "$PY" -m tasks.extract_natural_relation_reranker_s2_embeddings \
    --data-root "$DATA" --model "$MODEL" --tokenizer "$TOKENIZER" \
    --split development --selection "$CALIBRATION/selection.json" \
    --shard 1 --num-shards 2 --batch-size 16 \
    --output "$OUTPUT/embeddings/shard_1.npz" \
    > "$OUTPUT/embeddings/shard_1.log" 2>&1 &
  local pid1=$!
  wait "$pid0"; wait "$pid1"
}

run_scores() {
  local directory=$1
  local adapter=$2
  local extra=()
  if [[ -n "$adapter" ]]; then extra=(--adapter "$adapter"); fi
  CUDA_VISIBLE_DEVICES=0 "$PY" -m tasks.score_natural_relation_reranker_s2 \
    --data-root "$DATA" --model "$MODEL" --tokenizer "$TOKENIZER" \
    --split development --selection "$CALIBRATION/selection.json" \
    --shard 0 --num-shards 2 --pair-batch 16 --output "$directory/shard_0.jsonl" \
    "${extra[@]}" > "$directory/shard_0.log" 2>&1 &
  local pid0=$!
  CUDA_VISIBLE_DEVICES=1 "$PY" -m tasks.score_natural_relation_reranker_s2 \
    --data-root "$DATA" --model "$MODEL" --tokenizer "$TOKENIZER" \
    --split development --selection "$CALIBRATION/selection.json" \
    --shard 1 --num-shards 2 --pair-batch 16 --output "$directory/shard_1.jsonl" \
    "${extra[@]}" > "$directory/shard_1.log" 2>&1 &
  local pid1=$!
  wait "$pid0"; wait "$pid1"
}

run_embeddings
run_scores "$OUTPUT/selected" "$SELECTED_ADAPTER"
run_scores "$OUTPUT/zero_shot" ""
"$PY" -m tasks.evaluate_natural_relation_reranker_s2_development \
  --data-root "$DATA" --selection "$CALIBRATION/selection.json" \
  --selected-root "$OUTPUT/selected" --zero-shot-root "$OUTPUT/zero_shot" \
  --embedding-root "$OUTPUT/embeddings" --output "$OUTPUT/result.json" \
  > "$OUTPUT/result.log" 2>&1
"$PY" -m scripts.audit_natural_relation_reranker_s2_development \
  --data-root "$DATA" --selection "$CALIBRATION/selection.json" \
  --selected-root "$OUTPUT/selected" --zero-shot-root "$OUTPUT/zero_shot" \
  --embedding-root "$OUTPUT/embeddings" --result "$OUTPUT/result.json" \
  --output "$OUTPUT/audit.json" > "$OUTPUT/audit.log" 2>&1
date -Iseconds > "$OUTPUT/completed_at.txt"
"$PY" - "$OUTPUT/result.json" "$OUTPUT/terminal_status.txt" <<'PY'
import json, sys
open(sys.argv[2], "w").write(json.load(open(sys.argv[1]))["terminal_status"] + "\n")
PY
