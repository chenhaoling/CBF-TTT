#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/cbf_ttt_associative_binding_v1}
DATA=${DATA:-/home/ctj/cbf_ttt_dynamic_binding_v1/data}
MODEL=${MODEL:-/home/ctj/models/Qwen3-4B}
PYTHON=${PYTHON:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
TEST_PYTHON=${TEST_PYTHON:-/home/ctj/miniconda3/envs/cbf_ttt_verify_20260923/bin/python}

if [[ -e "$ROOT" ]]; then
  echo "Refusing to overwrite existing root: $ROOT" >&2
  exit 2
fi
mkdir -p "$ROOT"
cat > "$ROOT/RUN_STATUS.json" <<'JSON'
{"state":"running","stage":"tests"}
JSON

fail() {
  code=$?
  printf '{"state":"failed","exit_code":%s}\n' "$code" > "$ROOT/RUN_STATUS.json"
  exit "$code"
}
trap fail ERR

cd "$(dirname "$0")/.."
"$TEST_PYTHON" -m pytest -q \
  tests/test_cbf_associative_binding.py \
  tests/test_cbf_dynamic_binding.py \
  tests/test_cbf_candidate_ranking.py

cat > "$ROOT/RUN_STATUS.json" <<'JSON'
{"state":"running","stage":"training_and_evaluation"}
JSON

CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m tasks.train_cbf_associative_binding run \
  --data-root "$DATA" --output "$ROOT/anchor_twin" --model "$MODEL" \
  --arm anchor_twin --rounds 4 --train-groups 32 --dev-groups 8 &
pid0=$!
CUDA_VISIBLE_DEVICES=1 "$PYTHON" -m tasks.train_cbf_associative_binding run \
  --data-root "$DATA" --output "$ROOT/all_fact_permutation" --model "$MODEL" \
  --arm all_fact_permutation --rounds 4 --train-groups 32 --dev-groups 8 &
pid1=$!
wait "$pid0"
wait "$pid1"

"$PYTHON" -m tasks.train_cbf_associative_binding summarize --root "$ROOT"
"$PYTHON" -m scripts.audit_cbf_associative_binding --root "$ROOT" --data-root "$DATA"
git rev-parse HEAD > "$ROOT/git_commit.txt"
"$PYTHON" - "$ROOT" <<'PY'
import datetime,json,sys
from pathlib import Path
root=Path(sys.argv[1])
(root/'RUN_STATUS.json').write_text(json.dumps({
    'state':'completed',
    'updated_at':datetime.datetime.now().astimezone().isoformat(),
})+'\n')
PY
trap - ERR
