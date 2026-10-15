"""Audit the frozen scale-48 replay and its development-V2 dependency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.audit_qwen_causal_ledger_development_v2 import run as audit_development
from tasks.evaluate_qwen_causal_ledger_scale_rescue_v3 import PROTOCOL, SCALE
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


AUDIT_PROTOCOL = "qwen_causal_ledger_scale_rescue_v3_audit_v1"
EXPECTED_V2_FAILURE = "current_anchor_at_least_point_95"


def dependency_check(root, development_v2_root):
    root, prior = Path(root), Path(development_v2_root)
    dependency = json.loads((root / "development_v2_dependency.json").read_text())
    summary = json.loads((prior / "summary.json").read_text())
    audit = json.loads((prior / "execution_audit.json").read_text())
    failed = sorted(key for key, value in summary["gate"]["checks"].items() if not value)
    expected_hashes = {
        "summary.json": sha(prior / "summary.json"),
        "execution_audit.json": sha(prior / "execution_audit.json"),
        "data/manifest.json": sha(prior / "data/manifest.json"),
        "data/development.inputs.jsonl": sha(prior / "data/development.inputs.jsonl"),
        "data/development.targets.jsonl": sha(prior / "data/development.targets.jsonl"),
    }
    replay_hashes = {
        name: sha(root / name)
        for name in (
            "data/manifest.json",
            "data/development.inputs.jsonl",
            "data/development.targets.jsonl",
        )
    }
    return all((
        audit["passed"],
        not summary["gate"]["passed"],
        failed == [EXPECTED_V2_FAILURE],
        dependency["hashes"] == expected_hashes,
        all(replay_hashes[name] == expected_hashes[name] for name in replay_hashes),
    )), {
        "failed_v2_checks": failed,
        "registered_hashes": dependency["hashes"],
        "replay_data_hashes": replay_hashes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--development-v2-root", required=True)
    parser.add_argument("--residual-v2-root", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    # Reuse the independently implemented V2 row/state/bootstrap audit with
    # only the preregistered protocol, scale and terminal marker substituted.
    audit_args = argparse.Namespace(
        root=args.root,
        v2_root=args.residual_v2_root,
        tokenizer=args.tokenizer,
        output=args.output,
    )
    result = audit_development(
        audit_args,
        expected_protocol=PROTOCOL,
        expected_scale=SCALE,
        expected_terminal="completed_qwen_causal_ledger_scale_rescue_v3",
        audit_protocol=AUDIT_PROTOCOL,
    )
    dependency_ok, dependency = dependency_check(args.root, args.development_v2_root)
    result["checks"]["audited_v2_dependency_and_identical_data"] = dependency_ok
    result["development_v2_dependency"] = dependency
    result["passed"] = all(result["checks"].values())
    write_json(args.output, result)
    if not result["passed"]:
        raise RuntimeError("causal-ledger scale-rescue V3 audit failed")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
