"""Audit the one-time scale-48 confirmation and its passing V3 dependency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.audit_qwen_causal_ledger_development_v2 import run as audit_development
from tasks.build_qwen_causal_ledger_confirmation_v3 import (
    CONFIRMATION_SEED,
    load,
    PHASE,
)
from tasks.evaluate_qwen_causal_ledger_confirmation_v3 import PROTOCOL, SCALE
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


AUDIT_PROTOCOL = "qwen_causal_ledger_confirmation_v3_audit_v1"


def dependency_check(root, scale_rescue_root):
    root, prior = Path(root), Path(scale_rescue_root)
    dependency = json.loads((root / "scale_rescue_v3_dependency.json").read_text())
    summary = json.loads((prior / "summary.json").read_text())
    audit = json.loads((prior / "execution_audit.json").read_text())
    expected_hashes = {
        "summary.json": sha(prior / "summary.json"),
        "execution_audit.json": sha(prior / "execution_audit.json"),
    }
    manifest, pairs = load(root / "data")
    episode_ids = {inp["id"] for inp, _ in pairs}
    prior_ids = {
        json.loads(line)["id"]
        for line in (prior / "data/development.inputs.jsonl").read_text().splitlines()
        if line
    }
    return all((
        summary["gate"]["passed"],
        audit["passed"],
        dependency["hashes"] == expected_hashes,
        manifest["seed"] == CONFIRMATION_SEED,
        manifest["confirmation_generated"],
        not manifest["ood_generated"],
        len(episode_ids) == 896,
        not (episode_ids & prior_ids),
    )), {
        "registered_hashes": dependency["hashes"],
        "confirmation_seed": manifest["seed"],
        "confirmation_episodes": len(episode_ids),
        "development_confirmation_id_overlap": len(episode_ids & prior_ids),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--scale-rescue-root", required=True)
    parser.add_argument("--residual-v2-root", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
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
        expected_terminal="completed_qwen_causal_ledger_confirmation_v3",
        audit_protocol=AUDIT_PROTOCOL,
        loader=load,
        phase=PHASE,
        expected_confirmation_generated=True,
        expected_ood_generated=False,
    )
    dependency_ok, dependency = dependency_check(args.root, args.scale_rescue_root)
    result["checks"]["passing_scale_rescue_and_fresh_confirmation"] = dependency_ok
    result["scale_rescue_v3_dependency"] = dependency
    result["passed"] = all(result["checks"].values())
    write_json(args.output, result)
    if not result["passed"]:
        raise RuntimeError("causal-ledger confirmation V3 audit failed")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
