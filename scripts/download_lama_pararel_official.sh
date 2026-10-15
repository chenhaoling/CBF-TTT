#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/ctj/cbf_ttt_lama_pararel_sources_20261016}"
LAMA_REPO="${LAMA_REPO:-https://github.com/facebookresearch/LAMA.git}"
PARAREL_REPO="${PARAREL_REPO:-https://github.com/yanaiela/pararel.git}"
LAMA_DATA_URL="${LAMA_DATA_URL:-https://dl.fbaipublicfiles.com/LAMA/data.zip}"

if [[ -e "${ROOT}" ]]; then
  echo "refusing to overwrite ${ROOT}" >&2
  exit 2
fi

mkdir -p "${ROOT}/sources" "${ROOT}/downloads" "${ROOT}/metadata"
git clone --filter=blob:none "${LAMA_REPO}" "${ROOT}/sources/LAMA"
git clone --filter=blob:none "${PARAREL_REPO}" "${ROOT}/sources/pararel"
curl --fail --location --retry 5 --retry-delay 5 \
  --output "${ROOT}/downloads/lama_data.zip" "${LAMA_DATA_URL}"

git -C "${ROOT}/sources/LAMA" rev-parse HEAD > "${ROOT}/metadata/lama_commit.txt"
git -C "${ROOT}/sources/pararel" rev-parse HEAD > "${ROOT}/metadata/pararel_commit.txt"
sha256sum "${ROOT}/downloads/lama_data.zip" > "${ROOT}/metadata/lama_data.sha256"
find "${ROOT}/sources/LAMA" -type f -print0 | sort -z | xargs -0 sha256sum \
  > "${ROOT}/metadata/lama_repository_files.sha256"
find "${ROOT}/sources/pararel" -type f -print0 | sort -z | xargs -0 sha256sum \
  > "${ROOT}/metadata/pararel_repository_files.sha256"

python3 - "${ROOT}" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
def read(name):
    return (root / "metadata" / name).read_text().strip()

archive = root / "downloads" / "lama_data.zip"
manifest = {
    "protocol": "qwen_utility_semantic_record_sources_v1",
    "plan": "QWEN_UTILITY_SUPERVISED_SEMANTIC_RECORD_GATE_V1_PLAN.md",
    "content_parsed": False,
    "sources": {
        "lama": {
            "url": "https://github.com/facebookresearch/LAMA.git",
            "commit": read("lama_commit.txt"),
        },
        "pararel": {
            "url": "https://github.com/yanaiela/pararel.git",
            "commit": read("pararel_commit.txt"),
        },
        "lama_data": {
            "url": "https://dl.fbaipublicfiles.com/LAMA/data.zip",
            "bytes": archive.stat().st_size,
            "sha256": read("lama_data.sha256").split()[0],
        },
    },
    "forbidden_sources_loaded": False,
    "terminal_status": "sources_acquired_unparsed",
}
(root / "metadata" / "manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
)
PY

echo "sources acquired without parsing: ${ROOT}"
