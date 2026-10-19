#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEST=${1:?usage: bash data/preliminary_100m_v1/unpack.sh OUTPUT_DIRECTORY}

for command_name in sha256sum tar zstd python3; do
  command -v "$command_name" >/dev/null || {
    echo "missing required command: $command_name" >&2
    exit 2
  }
done

if [[ -e "$DEST" ]]; then
  echo "destination already exists: $DEST" >&2
  exit 3
fi

cd "$ROOT"
sha256sum -c PARTS_SHA256.txt

archive=$(mktemp "${TMPDIR:-/tmp}/cbf_ttt_preliminary_100m_v1.XXXXXX.tar.zst")
staging="${DEST}.partial"
if [[ -e "$staging" ]]; then
  echo "staging destination already exists: $staging" >&2
  rm -f "$archive"
  exit 3
fi
trap 'rm -f "$archive"; rm -rf "$staging"' EXIT

cat cbf_ttt_preliminary_100m_v1.tar.zst.part-* > "$archive"
expected=$(awk 'NR == 1 {print $1}' ARCHIVE_SHA256.txt)
actual=$(sha256sum "$archive" | awk '{print $1}')
if [[ "$actual" != "$expected" ]]; then
  echo "archive SHA256 mismatch: expected $expected, observed $actual" >&2
  exit 4
fi

mkdir -p "$staging"
tar --zstd -xf "$archive" -C "$staging"

python3 - "$staging" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
audit = json.loads((root / "independent_audit.json").read_text(encoding="utf-8"))
if audit.get("passed") is not True:
    raise SystemExit("included independent audit is not passing")
for stage in manifest["stages"]:
    for name, record in stage["artifacts"].items():
        path = root / name
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                hasher.update(chunk)
        digest = hasher.hexdigest()
        if path.stat().st_size != record["size_bytes"] or digest != record["sha256"]:
            raise SystemExit(f"artifact verification failed: {name}")
print(json.dumps({
    "profile": manifest["profile"],
    "records": manifest["totals"]["records"],
    "training_tokens": manifest["totals"]["training_tokens"],
    "audit_passed": True,
}, sort_keys=True))
PY

mv "$staging" "$DEST"
trap - EXIT
rm -f "$archive"
echo "dataset ready: $DEST"
