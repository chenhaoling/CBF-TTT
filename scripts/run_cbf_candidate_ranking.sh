#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/ctj/cbf_ttt_candidate_ranking_v1}"
SOURCE="${SOURCE:-/home/ctj/cbf_ttt_dynamic_binding_v1/formal}"
DATA="${DATA:-/home/ctj/cbf_ttt_dynamic_binding_v1/data}"
MODEL="${MODEL:-/home/ctj/models/Qwen3-4B}"
PYTHON="${PYTHON:-python3}"

if [[ -e "$ROOT" ]]; then
  echo "Refusing to overwrite $ROOT" >&2
  exit 1
fi
mkdir -p "$ROOT"

status() {
  "$PYTHON" -c 'import json,sys; from datetime import datetime; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps({"state":sys.argv[2],"updated_at":datetime.now().astimezone().isoformat()},indent=2)+"\n")' "$ROOT/RUN_STATUS.json" "$1"
}
status running
trap 'status failed' ERR

git rev-parse HEAD > "$ROOT/git_commit.txt"
"$PYTHON" -m unittest tests.test_cbf_dynamic_memory tests.test_cbf_dynamic_binding tests.test_cbf_candidate_ranking

CUDA_VISIBLE_DEVICES=0 "$PYTHON" tasks/evaluate_cbf_candidate_ranking.py evaluate \
  --source "$SOURCE" --data-root "$DATA" --output "$ROOT" --model "$MODEL" --arm joint_ce \
  > "$ROOT/joint_ce.log" 2>&1 &
left=$!
CUDA_VISIBLE_DEVICES=1 "$PYTHON" tasks/evaluate_cbf_candidate_ranking.py evaluate \
  --source "$SOURCE" --data-root "$DATA" --output "$ROOT" --model "$MODEL" --arm joint_binding \
  > "$ROOT/joint_binding.log" 2>&1 &
right=$!
wait "$left"
wait "$right"

"$PYTHON" tasks/evaluate_cbf_candidate_ranking.py summarize \
  --root "$ROOT" --formal-summary "$SOURCE/summary.json" > "$ROOT/summary.stdout.json"
"$PYTHON" scripts/audit_cbf_candidate_ranking.py \
  --root "$ROOT" --source "$SOURCE" --data-root "$DATA" \
  --formal-summary "$SOURCE/summary.json" > "$ROOT/audit.stdout.json"

trap - ERR
status completed
