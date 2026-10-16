#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/home/ctj/cbf_ttt_native_writer_distillation_v16_20261016}
SOURCE=${SOURCE:-/home/ctj/cbf_ttt_fastweight_content_v2_20261012}
MODEL=${MODEL:-/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt}
PYTHON=${PYTHON:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
if [[ -e "$ROOT" ]]; then echo "ROOT already exists: $ROOT" >&2; exit 2; fi

"$PYTHON" -m unittest tests.test_native_writer_distillation_v16 tests.test_final_writer_binding tests.test_cbf_event_writer
"$PYTHON" -m tasks.train_final_writer_binding build --source "$SOURCE" --root "$ROOT"
git rev-parse HEAD > "$ROOT/code_commit.txt"
date -Is > "$ROOT/started_at.txt"
trap 'code=$?; echo "$code" > "$ROOT/runner_exit_code.txt"; if (( code != 0 )); then date -Is > "$ROOT/failed_at.txt"; fi' EXIT

run_arm () {
  gpu=$1; arm=$2; scale=$3
  CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" -m tasks.train_native_writer_distillation_v16 \
    --root "$ROOT" --model "$MODEL" --output "$ROOT/$arm" --arm "$arm" \
    --teacher-scale "$scale" > "$ROOT/${arm}_train.log" 2>&1
  CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" -m tasks.train_final_writer_binding evaluate \
    --root "$ROOT" --model "$MODEL" --checkpoint "$ROOT/$arm/writer.pt" \
    --groups 8,9,10,11,12,13,14,15 --stage q --output "$ROOT/$arm/q.jsonl" \
    > "$ROOT/${arm}_q.log" 2>&1
  "$PYTHON" -m tasks.train_final_writer_binding summarize --root "$ROOT" --stage q \
    --inputs "$ROOT/$arm/q.jsonl" --output "$ROOT/$arm/q_summary.json"
}

run_arm 0 scale4 4 & p0=$!
run_arm 1 scale8 8 & p1=$!
status=0; wait "$p0" || status=1; wait "$p1" || status=1
if (( status != 0 )); then exit 1; fi

"$PYTHON" - "$ROOT" <<'PY'
import json, pathlib, sys
root=pathlib.Path(sys.argv[1]); out={"protocol":"native_writer_distillation_v16","arms":{}}
for arm in ("scale4","scale8"):
    train=json.loads((root/arm/"training_complete.json").read_text())
    q=json.loads((root/arm/"q_summary.json").read_text())
    out["arms"][arm]={"training":train,"q":q}
def passes(q):
    p=q["policies"]; c=q["comparisons"]
    return (p["correct"]["candidate_correct"]>=.5 and p["correct"]["eos_first"]<=.05
            and c["correct_vs_wrong_candidate_correct_gain"]["mean"]>=.25
            and c["correct_vs_zero_candidate_correct_gain"]["mean"]>=.25
            and all(c[f"correct_vs_{x}_nll_gain"]["mean"]>0
                    and c[f"correct_vs_{x}_nll_gain"]["ci95"][0]>0
                    for x in ("wrong","zero","random")))
out["gate"]={arm:passes(value["q"]) for arm,value in out["arms"].items()}
out["passed"]=any(out["gate"].values())
out["decision"]="fresh_source_confirmation" if out["passed"] else "stop_native_dense_writer"
(root/"summary.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,indent=2))
PY
echo completed_native_writer_distillation_v16 > "$ROOT/terminal_status.txt"
date -Is > "$ROOT/completed_at.txt"
