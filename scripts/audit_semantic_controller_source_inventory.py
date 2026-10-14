"""Rebuild and audit the post-V20 semantic-controller source inventory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tasks.build_semantic_controller_source_inventory import construct
from tasks.pack_cbf_event_warmup import sha


def read_jsonl(path):
    with Path(path).open() as source:
        return [json.loads(line) for line in source]


def audit(args):
    root = Path(args.inventory)
    manifest = json.loads((root / "manifest.json").read_text())
    expected_rows, expected_zsre, expected_excluded, expected_manifest = construct(
        args.counterfact, args.mquake, args.zsre_calibration, manifest["folds"],
        enforce_known_hashes=not args.allow_unknown_hashes)
    rows = read_jsonl(root / "semantic_rows.jsonl")
    zsre = read_jsonl(root / "zsre_calibration.jsonl")
    excluded = read_jsonl(root / "excluded_rows.jsonl")
    checks = {
        "semantic_rows_exact": rows == expected_rows,
        "zsre_calibration_exact": zsre == expected_zsre,
        "excluded_rows_exact": excluded == expected_excluded,
        "registered_protocol": manifest.get("protocol") == expected_manifest["protocol"],
        "source_hashes_exact": manifest.get("source_sha256") == expected_manifest["source_sha256"],
        "fold_counts_exact": manifest.get("fold_counts") == expected_manifest["fold_counts"],
        "exclusion_counts_exact": manifest.get("excluded_counts") == expected_manifest["excluded_counts"],
        "relation_disjoint": manifest.get("checks", {}).get("relation_disjoint") is True,
        "subject_disjoint": manifest.get("checks", {}).get("subject_disjoint") is True,
        "mquake_dev_unused": manifest.get("mquake_dev_used") is False,
        "zsre_dev_test_unloaded": manifest.get("zsre_dev_or_test_loaded") is False,
        "sealed_benchmarks_unloaded": (manifest.get("locomo_loaded") is False
                                        and manifest.get("longmemeval_loaded") is False),
        "model_outcomes_unloaded": manifest.get("model_outcomes_loaded") is False,
    }
    files = manifest.get("files_sha256", {})
    checks["file_hashes_exact"] = all((root / name).is_file() and sha(root / name) == digest
                                      for name, digest in files.items())
    relation_folds = json.loads((root / "relation_folds.json").read_text())
    checks["relation_fold_map_exact"] = relation_folds == {
        row["relation_id"]: row["outer_fold"] for row in rows}
    output = {"protocol": "audit_post_v20_semantic_controller_source_inventory_v1",
              "passed": all(checks.values()), "checks": checks,
              "manifest_sha256": sha(root / "manifest.json")}
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    if not output["passed"]:
        raise RuntimeError(json.dumps(output, indent=2))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counterfact", required=True)
    parser.add_argument("--mquake", required=True)
    parser.add_argument("--zsre-calibration", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-unknown-hashes", action="store_true")
    print(json.dumps(audit(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
