"""Independent audit for the calibration-only natural ZsRE forgetting pilot."""

import argparse
import json
from pathlib import Path

from tasks.pack_cbf_event_warmup import sha
from tasks.train_zsre_decomposed_scope_gate import GATE_SEEDS
from tasks.evaluate_zsre_natural_forgetting import (
    POLICIES, PROTOCOL, aggregate, comparisons, multi_seed,
)


def audit(root, data_root, source_root, scope_root):
    root, data_root, source, scope = map(Path, (root, data_root, source_root, scope_root))
    summary = json.loads((root / "summary.json").read_text())
    training = json.loads((root / "training.json").read_text())
    manifest = json.loads((root / "training_manifest.json").read_text())
    data_manifest = json.loads((data_root / "manifest.json").read_text())
    checks = {"protocol": summary["protocol"] == manifest["protocol"] == PROTOCOL,
              "data_hash": manifest["data_manifest_sha256"] == sha(data_root / "manifest.json"),
              "data_counts": data_manifest["counts"]["calibration"] == {
                  "subjects": 26, "chains": 26, "versions": 56,
                  "local_transitions": 30, "sessions": 4, "events": 56},
              "policies": tuple(manifest["policies"]) == POLICIES,
              "seeds": tuple(manifest["gate_seeds"]) == GATE_SEEDS,
              "calibration_only": manifest["split"] == "calibration_only"
                                  and not summary["dev_loaded"] and not summary["test_loaded"]
                                  and not summary["test_scored"] and not manifest["test_loaded"]
                                  and not manifest["test_scored"],
              "no_heldout_raw": not list(root.glob("dev_*.jsonl")) and not list(root.glob("test_*.jsonl")),
              "source_hashes": training["source_summary_sha256"] == sha(source / "summary.json")
                               and training["source_training_sha256"] == sha(source / "training.json")
                               and training["source_transforms_sha256"] == sha(source / "transforms.pt"),
              "scope_hashes": training["scope_training_sha256"] == sha(scope / "training.json")
                              and training["scope_checkpoint_sha256"] == sha(scope / "scope_gates.pt"),
              "features_present": bool(training["pair_features_sha256"]) and training["unique_pairs"] > 0}
    per_seed = {}
    for seed in GATE_SEEDS:
        per_seed[str(seed)] = {}
        for policy in POLICIES:
            path = root / f"calibration_{policy}_seed{seed}.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines() if line]
            checks[f"raw_hash_{policy}_{seed}"] = training["raw_rows_sha256"][path.name] == sha(path)
            checks[f"raw_identity_{policy}_{seed}"] = all(
                row["split"] == "calibration" and row["policy"] == policy and row["seed"] == seed
                for row in rows)
            per_seed[str(seed)][policy] = aggregate(rows)
    recomputed = multi_seed(per_seed)
    checks["summary_recomputed"] = recomputed == summary["calibration"]
    checks["comparison_recomputed"] = comparisons(recomputed) == summary["comparison"]
    return {"protocol": PROTOCOL, "audit_passed": all(checks.values()), "checks": checks,
            "dev_loaded": False, "test_loaded": False, "test_scored": False,
            "summary_sha256": sha(root / "summary.json"), "training_sha256": sha(root / "training.json"),
            "manifest_sha256": sha(root / "training_manifest.json"),
            "data_manifest_sha256": sha(data_root / "manifest.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True); parser.add_argument("--data-root", required=True)
    parser.add_argument("--source-root", required=True); parser.add_argument("--scope-root", required=True)
    parser.add_argument("--output", required=True); args = parser.parse_args()
    result = audit(args.root, args.data_root, args.source_root, args.scope_root)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result, indent=2))
    if not result["audit_passed"]: raise SystemExit(1)


if __name__ == "__main__": main()
