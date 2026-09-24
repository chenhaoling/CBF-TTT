"""Alternate equal-length FineWeb-Edu and LongCrawl64 rows for a small pilot."""

import argparse
import json
from itertools import zip_longest
from pathlib import Path


def read_rows(path):
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("content_split"), str) or not row["content_split"]:
                raise ValueError(f"{path}:{line_number} has empty content_split")
            yield row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fineweb", required=True)
    parser.add_argument("--longcrawl", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    if output.resolve() in (Path(args.fineweb).resolve(), Path(args.longcrawl).resolve()):
        parser.error("output must differ from both inputs")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".incomplete")
    count = 0
    with temporary.open("w", encoding="utf-8") as sink:
        for first, second in zip_longest(read_rows(args.fineweb), read_rows(args.longcrawl)):
            if first is None or second is None:
                raise ValueError("equal-token mix requires identical record counts")
            for source_name, row in (("fineweb_edu", first), ("longcrawl64", second)):
                sink.write(json.dumps({"content_split": row["content_split"],
                                       "source": source_name}, ensure_ascii=False) + "\n")
            count += 1
    if count == 0:
        raise ValueError("source files contain no records")
    temporary.replace(output)
    print(json.dumps({"output": str(output), "records": 2 * count,
                      "source_records": {"fineweb_edu": count,
                                         "longcrawl64": count}}))


if __name__ == "__main__":
    main()
