"""Audit nested folds and relation/source balancing for semantic training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tasks.pack_cbf_event_warmup import sha
from tasks.prepare_semantic_controller_balancing import prepare, read_jsonl


def audit(args):
    inventory, root = Path(args.inventory), Path(args.balancing)
    manifest = json.loads((root / "manifest.json").read_text())
    rows = read_jsonl(inventory / "semantic_rows.jsonl")
    expected, summaries = prepare(rows, manifest["folds"])
    checks = {
        "protocol": manifest.get("protocol") == "post_v20_semantic_controller_nested_balancing_v2",
        "inventory_hash": manifest.get("inventory_manifest_sha256") == sha(inventory / "manifest.json"),
        "semantic_rows_hash": manifest.get("semantic_rows_sha256") == sha(inventory / "semantic_rows.jsonl"),
        "fold_summaries": manifest.get("fold_summaries") == summaries,
        "outcomes_unused": manifest.get("model_outcomes_loaded") is False,
        "sealed_benchmarks_unused": manifest.get("sealed_benchmarks_loaded") is False,
    }
    exact, hashes = True, True
    for outer, expected_rows in expected.items():
        name = f"outer_{outer}.jsonl"
        exact &= read_jsonl(root / name) == expected_rows
        hashes &= manifest.get("files_sha256", {}).get(name) == sha(root / name)
    checks["fold_rows_exact"] = exact; checks["file_hashes_exact"] = hashes
    checks["all_resource_gates"] = all(
        item["checks"]["resource_gate_passed"] for item in summaries.values())
    result = {"protocol": "audit_post_v20_semantic_controller_nested_balancing_v2",
              "passed": all(checks.values()), "checks": checks,
              "manifest_sha256": sha(root / "manifest.json")}
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not result["passed"]:
        raise RuntimeError(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--balancing", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(audit(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
