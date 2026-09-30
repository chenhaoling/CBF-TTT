#!/usr/bin/env bash
# Run from the repository root, inside the existing CUDA conda environment.
set -euo pipefail
ROOT=${ROOT:-/home/ctj/cbf_ttt_paired_fact_20260930}
MODEL=${MODEL:-/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt}
TOKENIZER=${TOKENIZER:-/home/ctj/models/Qwen3-4B}
PYTHON=${PYTHON:-python}
git rev-parse HEAD > "$ROOT/code_commit.txt"
"$PYTHON" -m tasks.build_cbf_paired_facts --data "$ROOT/documents.jsonl" --output "$ROOT/episodes.jsonl" --tokenizer "$TOKENIZER"
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m tasks.cbf_paired_writer sanity --model "$MODEL" --data "$ROOT/episodes.jsonl" --features "$ROOT/features" --output "$ROOT/sanity" > "$ROOT/sanity.log" 2>&1
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m tasks.cbf_paired_writer extract --model "$MODEL" --data "$ROOT/episodes.jsonl" --features "$ROOT/features" --shards 2 --shard 0 > "$ROOT/extract0.log" 2>&1 &
pid0=$!
CUDA_VISIBLE_DEVICES=1 "$PYTHON" -m tasks.cbf_paired_writer extract --model "$MODEL" --data "$ROOT/episodes.jsonl" --features "$ROOT/features" --shards 2 --shard 1 > "$ROOT/extract1.log" 2>&1 &
pid1=$!
failed=0
wait "$pid0" || failed=1
wait "$pid1" || failed=1
[[ "$failed" == 0 ]] || exit 1
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m tasks.cbf_paired_writer train --model "$MODEL" --data "$ROOT/episodes.jsonl" --features "$ROOT/features" --objective paired --smoke --output "$ROOT/smoke" > "$ROOT/smoke.log" 2>&1
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m tasks.cbf_paired_writer train --model "$MODEL" --data "$ROOT/episodes.jsonl" --features "$ROOT/features" --objective paired --output "$ROOT/train_paired" > "$ROOT/paired.log" 2>&1 &
pid0=$!
CUDA_VISIBLE_DEVICES=1 "$PYTHON" -m tasks.cbf_paired_writer train --model "$MODEL" --data "$ROOT/episodes.jsonl" --features "$ROOT/features" --objective nll --output "$ROOT/train_nll" > "$ROOT/nll.log" 2>&1 &
pid1=$!
failed=0
wait "$pid0" || failed=1
wait "$pid1" || failed=1
[[ "$failed" == 0 ]] || exit 1
"$PYTHON" - "$ROOT" <<'PY'
import json
import pathlib
import sys
root = pathlib.Path(sys.argv[1])
paired = json.loads((root/'train_paired/selection.json').read_text())
nll = json.loads((root/'train_nll/selection.json').read_text())
status = {'paired_passed_dev': paired['passed_dev_gate'], 'nll_passed_dev': nll['passed_dev_gate'],
          'test_authorized_by_protocol': paired['passed_dev_gate']}
(root/'stage_status.json').write_text(json.dumps(status, indent=2)+'\n')
print(json.dumps(status))
PY
if "$PYTHON" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["passed_dev_gate"] else 1)' "$ROOT/train_paired/selection.json"; then
    CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m tasks.cbf_paired_writer test --model "$MODEL" --data "$ROOT/episodes.jsonl" --features "$ROOT/features" --paired-dir "$ROOT/train_paired" --nll-dir "$ROOT/train_nll" --output "$ROOT/test" > "$ROOT/test.log" 2>&1
fi
date -u +%FT%TZ > "$ROOT/completed.txt"
