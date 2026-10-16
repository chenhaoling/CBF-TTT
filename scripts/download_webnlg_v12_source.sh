#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/data/webnlg_v16_official_20261016}
REPOSITORY=${REPOSITORY:-https://github.com/ThiagoCF05/webnlg.git}
PYTHON_BIN=${PYTHON_BIN:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}

if [[ -e "$ROOT" ]]; then echo "Refusing to overwrite $ROOT" >&2; exit 2; fi
mkdir -p "$ROOT"
git clone --quiet "$REPOSITORY" "$ROOT/repository"
git -C "$ROOT/repository" rev-parse HEAD > "$ROOT/repository_commit.txt"
date --iso-8601=seconds > "$ROOT/downloaded_at.txt"

"$PYTHON_BIN" - "$ROOT" <<'PY'
from pathlib import Path
import hashlib, json, sys
root = Path(sys.argv[1]); repo = root / "repository"
candidates = [p for p in repo.rglob("*") if p.is_file()
              and "v1.6" in p.as_posix() and "/en/" in p.as_posix()]
if not candidates:
    raise SystemExit("no English WebNLG v1.6 files found")
files = {}
for path in sorted(candidates):
    data = path.read_bytes()
    files[str(path.relative_to(repo))] = {
        "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
    }
manifest = {
    "protocol": "qwen_utility_record_controller_v12_webnlg_source_acquisition",
    "repository": "https://github.com/ThiagoCF05/webnlg.git",
    "repository_commit": (root / "repository_commit.txt").read_text().strip(),
    "english_v16_files": files,
    "xml_or_json_parsed": False,
    "model_loaded": False,
    "sealed_benchmarks_loaded": False,
}
(root / "source_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print(json.dumps({"commit": manifest["repository_commit"],
                  "files": len(files),
                  "bytes": sum(row["bytes"] for row in files.values())},
                 indent=2, sort_keys=True))
PY

echo completed_qwen_utility_record_controller_v12_webnlg_source_acquisition \
  > "$ROOT/terminal_status.txt"
echo 0 > "$ROOT/runner_exit_code.txt"
