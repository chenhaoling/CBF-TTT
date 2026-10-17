#!/usr/bin/env bash
set -Eeuo pipefail

REPO=/home/ctj/in_place_ttt_official_be232482
PYTHON_BIN=/home/ctj/miniconda3/envs/cbf_ttt_verify_20260923/bin/python
TOKENIZER=/home/ctj/models/Qwen3-4B
ROOT=/home/ctj/in_place_ttt_public_corpus_inventory_20261018
LOG=/home/ctj/in_place_ttt_public_corpus_inventory_20261018.log
STATUS="$ROOT/runner_status.json"

test ! -e "$ROOT"
mkdir "$ROOT"
exec > >(tee -a "$LOG") 2>&1
STARTED="$(date -Iseconds)"; export STARTED
"$PYTHON_BIN" - "$STATUS" <<'PY'
import json,os,sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({"status":"running","started_at":os.environ["STARTED"]},indent=2)+"\n")
PY

finish() {
  local rc=$?
  trap - EXIT
  INVENTORY_RC="$rc" ENDED="$(date -Iseconds)" "$PYTHON_BIN" - "$STATUS" <<'PY'
import json,os,sys
from pathlib import Path
p=Path(sys.argv[1]); value=json.loads(p.read_text())
value.update({"status":"completed" if int(os.environ["INVENTORY_RC"])==0 else "failed",
              "exit_code":int(os.environ["INVENTORY_RC"]),"ended_at":os.environ["ENDED"]})
q=p.with_name(p.name+".partial");q.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n");q.replace(p)
PY
  exit "$rc"
}
trap finish EXIT

cd "$REPO"
"$PYTHON_BIN" scripts/inventory_in_place_ttt_public_corpus.py \
  --tokenizer "$TOKENIZER" \
  --output "$ROOT/inventory.json"
"$PYTHON_BIN" scripts/audit_in_place_ttt_public_corpus_inventory.py \
  --inventory "$ROOT/inventory.json" \
  --tokenizer "$TOKENIZER" \
  --output "$ROOT/independent_audit.json"
"$PYTHON_BIN" - "$ROOT" <<'PY'
import hashlib,json,sys
from pathlib import Path
r=Path(sys.argv[1]);inv=json.loads((r/'inventory.json').read_text());audit=json.loads((r/'independent_audit.json').read_text())
if not audit['passed']: raise SystemExit('inventory audit failed')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
summary={"protocol":"in_place_ttt_public_corpus_inventory_v1_summary","passed":True,
         "inventory_sha256":sha(r/'inventory.json'),"audit_sha256":sha(r/'independent_audit.json'),
         "sources":{name:{"files":value['file_count'],"total_size_bytes":value['total_size_bytes'],
                           "ordered_files_sha256":value['ordered_files_sha256']}
                    for name,value in inv['sources'].items()},"resources":inv['resources']}
(r/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
print(json.dumps(summary,sort_keys=True))
PY
