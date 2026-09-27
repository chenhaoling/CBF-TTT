"""Check whether selected paper titles occur in the 1B continued-pretraining corpus."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path


def audit(metadata: Path, corpus: Path) -> dict:
    source = json.loads(metadata.read_text(encoding="utf-8"))
    entries = source["accepted"]
    titles = [" ".join(row["title"].split()) for row in entries]
    if len(titles) != len(set(title.casefold() for title in titles)) or any(len(title) < 12 for title in titles):
        raise ValueError("expected distinct, nontrivial titles")
    with tempfile.TemporaryDirectory() as directory:
        patterns = Path(directory) / "titles.txt"
        patterns.write_text("\n".join(titles) + "\n", encoding="utf-8")
        result = subprocess.run(["rg", "--ignore-case", "--fixed-strings", "--only-matching",
                                 "--no-filename", "--file", str(patterns), str(corpus)],
                                capture_output=True, text=True, check=False)
    if result.returncode not in (0, 1):
        raise RuntimeError(f"title scan failed: {result.stderr[:200]}")
    matches = {line.casefold() for line in result.stdout.splitlines() if line.strip()}
    matched = [row["source_id"] for row, title in zip(entries, titles)
               if title.casefold() in matches]
    return {"protocol": "postcutoff_title_audit_v1", "documents": len(entries),
            "matched_titles": len(matched), "matched_source_ids": matched,
            "corpus_path": str(corpus),
            "note": "Exact case-insensitive title scan; absence does not prove full-text disjointness."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.metadata, args.corpus)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
