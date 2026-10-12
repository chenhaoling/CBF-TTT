"""Independent aggregation and release audit for ZsRE contextual readout."""

import argparse
import json
from pathlib import Path

from tasks.pack_cbf_event_warmup import sha
from tasks.evaluate_zsre_contextual_readout import (
    REGULARIZERS, SEEDS, UNKNOWN_GRID, aggregate, gate, select_config, summarize,
)


def audit(root, data_root):
    root, data_root = Path(root), Path(data_root)
    summary = json.loads((root / "summary.json").read_text())
    training = json.loads((root / "training.json").read_text())
    manifest = json.loads((root / "training_manifest.json").read_text())
    data = json.loads((data_root / "manifest.json").read_text())
    subject_sets = {name: set() for name in ("calibration", "dev", "test")}
    for split in subject_sets:
        path = data_root / f"{split}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        subject_sets[split] = {fact["subject"] for row in rows for fact in row["facts"]}
    checks = {
        "protocol": summary["protocol"] == manifest["protocol"] == data["protocol"]
                    == "zsre_contextual_readout_v1",
        "data_hash": manifest["data_manifest_sha256"] == sha(data_root / "manifest.json"),
        "source_hash": data["source_sha256"]
                       == "0e0214dda853a906ef02bef58c7ce2978af5a962e91fc4a10827e0125459905c",
        "subject_counts": {name: len(values) for name, values in subject_sets.items()}
                          == {"calibration": 267, "dev": 267, "test": 534},
        "subject_disjoint": not any(subject_sets[a] & subject_sets[b]
                                    for a in subject_sets for b in subject_sets if a < b),
        "seeds": tuple(manifest["seeds"]) == SEEDS,
        "regularizers": tuple(manifest["regularizers"]) == REGULARIZERS,
        "unknown_grid": tuple(manifest["unknown_grid"]) == UNKNOWN_GRID,
        "selection": select_config(training["calibration_grid"])
                     == training["selected_config"] == summary["selected_config"],
        "transform_hash": training["transforms_sha256"] == sha(root / "transforms.pt"),
    }
    dev_seed = {}
    for seed in SEEDS:
        path = root / f"dev_seed{seed}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        checks[f"dev_hash_{seed}"] = training["raw_rows_sha256"][path.name] == sha(path)
        checks[f"dev_identity_{seed}"] = all(row["split"] == "dev" and row["seed"] == seed for row in rows)
        dev_seed[str(seed)] = aggregate(rows)
    rebuilt_dev = summarize(dev_seed)
    checks["dev_recomputed"] = rebuilt_dev == summary["dev"]
    checks["gate_recomputed"] = gate(rebuilt_dev) == summary["dev_gate"]
    test_exists = all((root / f"test_seed{seed}.jsonl").exists() for seed in SEEDS)
    checks["test_lock"] = test_exists == bool(summary["dev_gate"]["passed"]) \
                          == bool(summary["test_scored"]) == bool(manifest["test_scored"])
    if test_exists:
        test_seed = {}
        for seed in SEEDS:
            path = root / f"test_seed{seed}.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines() if line]
            checks[f"test_hash_{seed}"] = training["raw_rows_sha256"][path.name] == sha(path)
            checks[f"test_identity_{seed}"] = all(row["split"] == "test" and row["seed"] == seed for row in rows)
            test_seed[str(seed)] = aggregate(rows)
        checks["test_recomputed"] = summarize(test_seed) == summary["test"]
        checks["test_embeddings_present"] = training["test_embedding_hashes"] is not None
    else:
        checks["test_embeddings_absent"] = training["test_embedding_hashes"] is None
    result = {"protocol": "zsre_contextual_readout_v1", "audit_passed": all(checks.values()),
              "checks": checks, "dev_gate_passed": summary["dev_gate"]["passed"],
              "stage_passed": summary["stage_passed"], "test_scored": summary["test_scored"],
              "summary_sha256": sha(root / "summary.json"),
              "training_sha256": sha(root / "training.json"),
              "manifest_sha256": sha(root / "training_manifest.json"),
              "data_manifest_sha256": sha(data_root / "manifest.json")}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True); parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True); args = parser.parse_args()
    result = audit(args.root, args.data_root)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["audit_passed"]: raise SystemExit(1)


if __name__ == "__main__": main()
