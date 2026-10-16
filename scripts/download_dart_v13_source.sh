#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/data/dart_v111_official_20261016}
REPOSITORY=${REPOSITORY:-https://github.com/Yale-LILY/dart.git}
PYTHON_BIN=${PYTHON_BIN:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}

if [[ -e "$ROOT" ]]; then
  echo "Refusing to overwrite $ROOT" >&2
  exit 2
fi

mkdir -p "$ROOT"
git clone --quiet "$REPOSITORY" "$ROOT/repository"
git -C "$ROOT/repository" rev-parse HEAD > "$ROOT/repository_commit.txt"
date --iso-8601=seconds > "$ROOT/downloaded_at.txt"

"$PYTHON_BIN" - "$ROOT" <<'PY'
from pathlib import Path
import hashlib
import json
import sys

root = Path(sys.argv[1])
repo = root / "repository"
data_root = repo / "data" / "v1.1.1"
if not data_root.is_dir():
    raise SystemExit("missing official DART data/v1.1.1 directory")

candidates = [path for path in data_root.rglob("*") if path.is_file()]
if not candidates:
    raise SystemExit("no official DART v1.1.1 files found")

files = {}
for path in sorted(candidates):
    payload = path.read_bytes()
    files[str(path.relative_to(repo))] = {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }

manifest = {
    "protocol": "qwen_utility_record_controller_v13_dart_source_acquisition",
    "repository": "https://github.com/Yale-LILY/dart.git",
    "repository_commit": (root / "repository_commit.txt").read_text().strip(),
    "dart_v111_files": files,
    "json_or_xml_parsed": False,
    "model_loaded": False,
    "sealed_benchmarks_loaded": False,
}
(root / "source_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
)
print(json.dumps({
    "commit": manifest["repository_commit"],
    "files": len(files),
    "bytes": sum(row["bytes"] for row in files.values()),
}, indent=2, sort_keys=True))
PY

echo completed_qwen_utility_record_controller_v13_dart_source_acquisition \
  > "$ROOT/terminal_status.txt"
echo 0 > "$ROOT/runner_exit_code.txt"
