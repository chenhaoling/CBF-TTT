#!/usr/bin/env python3
"""Summarize one 2x5090 100M-tier RULER context from raw output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.summarize_in_place_ttt_level_b_ruler import (
        summarize as summarize_common,
    )
except ModuleNotFoundError:  # Direct `python scripts/<name>.py` execution.
    from summarize_in_place_ttt_level_b_ruler import summarize as summarize_common


PROTOCOL = "in_place_ttt_2x5090_ruler_summary_v1"


def summarize(root: Path) -> dict:
    result = summarize_common(root)
    result["protocol"] = PROTOCOL
    result["baseline_tier"] = "2x5090_100m"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    result = summarize(args.root.expanduser().resolve())
    partial.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    partial.replace(output)
    print(json.dumps({"output": str(output), "completed": result["completed"]},
                     sort_keys=True))
    raise SystemExit(0 if result["completed"] else 1)


if __name__ == "__main__":
    main()
