#!/usr/bin/env bash
set -Eeuo pipefail

REPO=/home/ctj/in_place_ttt_official_be232482
PYTHON_BIN=/home/ctj/miniconda3/envs/cbf_ttt_verify_20260923/bin/python
TOKENIZER=/home/ctj/models/Qwen3-4B
OUTPUT_ROOT=/home/ctj/in_place_ttt_public_data_pilot_20261017_r2
LONGCRAWL_INPUT=/home/ctj/data/cbf_ttt_pilot/longcrawl64/train/0-of-256.parquet
STATUS_PATH="$OUTPUT_ROOT/runner_status.json"
LOG_PATH=/home/ctj/in_place_ttt_public_data_pilot_20261017_r2.log

test ! -e "$OUTPUT_ROOT"
mkdir "$OUTPUT_ROOT"
exec > >(tee -a "$LOG_PATH") 2>&1

PILOT_STARTED_AT="$(date -Iseconds)"
export PILOT_STARTED_AT
"$PYTHON_BIN" - "$STATUS_PATH" <<'PY'
import json, os, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "status": "running", "started_at": os.environ["PILOT_STARTED_AT"]
}, indent=2) + "\n")
PY

finish() {
    local rc=$?
    trap - EXIT
    PILOT_EXIT_CODE="$rc" PILOT_ENDED_AT="$(date -Iseconds)" \
        "$PYTHON_BIN" - "$STATUS_PATH" <<'PY'
import json, os, sys
from pathlib import Path
path = Path(sys.argv[1])
old = json.loads(path.read_text()) if path.exists() else {}
old.update({
    "status": "completed" if int(os.environ["PILOT_EXIT_CODE"]) == 0 else "failed",
    "exit_code": int(os.environ["PILOT_EXIT_CODE"]),
    "ended_at": os.environ["PILOT_ENDED_AT"],
})
partial = path.with_name(path.name + ".partial")
partial.write_text(json.dumps(old, indent=2, sort_keys=True) + "\n")
partial.replace(path)
PY
    exit "$rc"
}
trap finish EXIT

export HF_HUB_ETAG_TIMEOUT=30
export HF_HUB_DOWNLOAD_TIMEOUT=120
cd "$REPO"

echo "[$(date -Iseconds)] FineWeb-Edu build"
"$PYTHON_BIN" scripts/build_in_place_ttt_public_data_pilot.py \
    --source fineweb \
    --tokenizer "$TOKENIZER" \
    --output-root "$OUTPUT_ROOT/fineweb" \
    --max-records 8 \
    --seq-len 32768 \
    --max-documents 10000

echo "[$(date -Iseconds)] FineWeb-Edu independent audit"
"$PYTHON_BIN" scripts/audit_in_place_ttt_public_data_pilot.py \
    --root "$OUTPUT_ROOT/fineweb" \
    --tokenizer "$TOKENIZER"

echo "[$(date -Iseconds)] LongCrawl64 build"
"$PYTHON_BIN" scripts/build_in_place_ttt_public_data_pilot.py \
    --source longcrawl \
    --longcrawl-input "$LONGCRAWL_INPUT" \
    --tokenizer "$TOKENIZER" \
    --output-root "$OUTPUT_ROOT/longcrawl" \
    --max-records 8 \
    --seq-len 32768 \
    --max-documents 10000

echo "[$(date -Iseconds)] LongCrawl64 independent audit"
"$PYTHON_BIN" scripts/audit_in_place_ttt_public_data_pilot.py \
    --root "$OUTPUT_ROOT/longcrawl" \
    --tokenizer "$TOKENIZER"

"$PYTHON_BIN" - "$OUTPUT_ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
summary = {}
for source in ("fineweb", "longcrawl"):
    manifest = json.loads((root / source / "manifest.json").read_text())
    audit = json.loads((root / source / "independent_audit.json").read_text())
    if not audit["passed"]:
        raise SystemExit(f"{source} audit did not pass")
    summary[source] = {
        "records": manifest["packing"]["records"],
        "training_tokens": manifest["packing"]["total_training_tokens"],
        "wall_seconds": manifest["resources"]["wall_seconds"],
        "peak_host_rss_mib": manifest["resources"]["peak_host_rss_mib"],
        "manifest_sha256": audit["bindings"]["manifest_sha256"],
        "audit_sha256": __import__("hashlib").sha256(
            (root / source / "independent_audit.json").read_bytes()
        ).hexdigest(),
    }
path = root / "summary.json"
path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, sort_keys=True))
PY

echo "[$(date -Iseconds)] pilot completed"
