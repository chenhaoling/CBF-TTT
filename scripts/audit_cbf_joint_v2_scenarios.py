"""Audit held-out target strings in v2 pretokenized context and gap text."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def audit(data: Path, tokenizer) -> dict:
    counts = Counter()
    seen = set()
    with data.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["objective"] != "joint_v2" or row["id"] in seen:
                raise ValueError("expected unique joint_v2 scenarios")
            seen.add(row["id"])
            counts["scenarios"] += 1
            for future in row["futures"]:
                visible = tokenizer.decode(row["context_ids"] + future["continuation_ids"],
                                           skip_special_tokens=True)
                for query in future["queries"]:
                    if query["kind"] == "neutral_heldout":
                        counts["neutral_queries"] += 1
                        continue
                    target = tokenizer.decode(query["answer_ids"], skip_special_tokens=True).strip()
                    if not target or target in visible:
                        raise ValueError(f"held-out target visible in {row['id']} gap={future['gap_chunks']}")
                    counts["rule_queries_without_visible_target"] += 1
    if not seen:
        raise ValueError("no scenarios found")
    return dict(counts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    result = audit(args.data, tokenizer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
