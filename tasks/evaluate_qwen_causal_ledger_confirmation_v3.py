"""Evaluate the one-time frozen scale-48 causal-ledger confirmation."""

from __future__ import annotations

import argparse
import json

from tasks.build_qwen_causal_ledger_confirmation_v3 import load, PHASE
from tasks.evaluate_qwen_causal_ledger_development_v2 import collect, summarize


PROTOCOL = "qwen_causal_ledger_confirmation_v3_v1"
SCALE = 48.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument("--data-root", required=True)
    collect_parser.add_argument("--model", required=True)
    collect_parser.add_argument("--tokenizer", required=True)
    collect_parser.add_argument("--shard", type=int, required=True)
    collect_parser.add_argument("--num-shards", type=int, default=2)
    collect_parser.add_argument("--output", required=True)
    collect_parser.add_argument("--resource-output", required=True)
    summary_parser = sub.add_parser("summarize")
    summary_parser.add_argument("--root", required=True)
    summary_parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "collect":
        result = collect(
            args, scale=SCALE, protocol=PROTOCOL, loader=load, phase=PHASE,
            confirmation_generated=True, ood_generated=False,
        )
    else:
        result = summarize(
            args, scale=SCALE, protocol=PROTOCOL,
            confirmation_generated=True, ood_generated=False,
            passing_next_stage="independent_confirmation_audit_before_native_bridge",
            failure_next_stage="stop_before_native_bridge",
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
