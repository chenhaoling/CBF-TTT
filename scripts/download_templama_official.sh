#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/cbf_ttt_templama_sources_20261016}
REPOSITORY=${REPOSITORY:-https://github.com/google-research/language.git}
DATA_BASE=${DATA_BASE:-https://storage.googleapis.com/gresearch/templama}

if [[ -e "$ROOT" ]]; then
  echo "Refusing to overwrite $ROOT" >&2
  exit 2
fi

mkdir -p "$ROOT/sources" "$ROOT/downloads" "$ROOT/metadata"
git clone --filter=blob:none --no-checkout "$REPOSITORY" \
  "$ROOT/sources/google-research-language"
git -C "$ROOT/sources/google-research-language" sparse-checkout init --cone
git -C "$ROOT/sources/google-research-language" sparse-checkout set language/templama
git -C "$ROOT/sources/google-research-language" checkout

for split in train val test; do
  curl --fail --location --retry 5 --retry-delay 5 \
    --output "$ROOT/downloads/$split.json" "$DATA_BASE/$split.json"
done

git -C "$ROOT/sources/google-research-language" rev-parse HEAD \
  > "$ROOT/metadata/repository_commit.txt"
find "$ROOT/sources/google-research-language/language/templama" -type f -print0 \
  | sort -z | xargs -0 sha256sum > "$ROOT/metadata/repository_files.sha256"
for split in train val test; do
  sha256sum "$ROOT/downloads/$split.json" \
    > "$ROOT/metadata/$split.sha256"
done

python3 - "$ROOT" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])

def text(name):
    return (root / "metadata" / name).read_text().strip()

downloads = {}
for split in ("train", "val", "test"):
    path = root / "downloads" / f"{split}.json"
    downloads[split] = {
        "url": f"https://storage.googleapis.com/gresearch/templama/{split}.json",
        "bytes": path.stat().st_size,
        "sha256": text(f"{split}.sha256").split()[0],
    }

manifest = {
    "protocol": "qwen_utility_record_controller_v2_templama_sources",
    "plan": "QWEN_UTILITY_RECORD_CONTROLLER_V2_CROSS_SOURCE_PLAN.md",
    "content_parsed": False,
    "sources": {
        "repository": {
            "url": "https://github.com/google-research/language.git",
            "subdirectory": "language/templama",
            "commit": text("repository_commit.txt"),
        },
        "downloads": downloads,
    },
    "v1_confirmation_loaded": False,
    "forbidden_sources_loaded": False,
    "terminal_status": "templama_sources_acquired_unparsed",
}
(root / "metadata" / "manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
)
PY

echo "TempLAMA sources acquired without parsing: $ROOT"
