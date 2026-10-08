#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/home/ctj/cbf_ttt_multidecision_20261008}
SOURCE=${SOURCE:-/home/ctj/cbf_ttt_selective_forgetting_20261008_r3}
PYTHON=${PYTHON:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
MODEL=${MODEL:-/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt}
mkdir "$ROOT"
git rev-parse HEAD > "$ROOT/code_commit.txt"
date -Is > "$ROOT/started_at.txt"
"$PYTHON" -m unittest tests.test_cbf_multidecision tests.test_cbf_selective tests.test_cbf_ttt > "$ROOT/tests.log" 2>&1
"$PYTHON" -m tasks.cbf_multidecision build --data "$SOURCE/scenes.jsonl" --output "$ROOT/design.json" > "$ROOT/build.log" 2>&1
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m tasks.cbf_multidecision collect \
  --model "$MODEL" --data "$SOURCE/scenes.jsonl" --design "$ROOT/design.json" \
  --reference "$SOURCE/labels_0.jsonl" "$SOURCE/labels_1.jsonl" \
  --output "$ROOT/smoke.jsonl" --shard 0 --smoke > "$ROOT/smoke.log" 2>&1
"$PYTHON" - "$ROOT" <<'PY' > "$ROOT/estimate.json"
import json,statistics,sys
from pathlib import Path
p=Path(sys.argv[1]); d=json.loads((p/'design.json').read_text()); r=json.loads((p/'smoke.jsonl').read_text())
profiles=[x for x in list(r['results'].values())+list(r['controls'].values()) if not x['reused']]+r['reference_checks']
new_paths=sum((len(v['all_plans'])-27+5)*4 for v in d['groups'].values())
seconds=statistics.fmean(x['seconds'] for x in profiles)
estimate=new_paths*seconds/2*1.25
peak=max(x['peak_allocated_gib'] for x in profiles)
print(json.dumps({'new_gpu_trajectories_including_bridges':new_paths,'mean_smoke_seconds':seconds,
                  'estimated_two_gpu_seconds_with_25pct_margin':estimate,'smoke_peak_allocated_gib':peak,
                  'passed':estimate<=5400 and peak<30},indent=2))
assert estimate<=5400 and peak<30, 'predeclared resource budget failed; do not auto-scale'
PY
pids=()
for gpu in 0 1; do
  CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" -m tasks.cbf_multidecision collect \
    --model "$MODEL" --data "$SOURCE/scenes.jsonl" --design "$ROOT/design.json" \
    --reference "$SOURCE/labels_0.jsonl" "$SOURCE/labels_1.jsonl" \
    --output "$ROOT/rows_${gpu}.jsonl" --shard "$gpu" > "$ROOT/collect_${gpu}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if (( status != 0 )); then exit 1; fi
"$PYTHON" -m tasks.cbf_multidecision summarize --design "$ROOT/design.json" \
  --inputs "$ROOT/rows_0.jsonl" "$ROOT/rows_1.jsonl" --output "$ROOT/summary.json" > "$ROOT/summary.log" 2>&1
date -Is > "$ROOT/completed_at.txt"
