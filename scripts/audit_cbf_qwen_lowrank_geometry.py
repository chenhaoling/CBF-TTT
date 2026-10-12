"""Independent aggregation and protocol audit for Qwen memory geometry."""

import argparse
import json
from pathlib import Path

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_qwen_lowrank_forgetting import SEEDS
from tasks.diagnose_cbf_qwen_lowrank_geometry import (
    LOADS, REGULARIZERS, VARIANT_ORDER, aggregate_readability,
    mean_seed_summary, readability_gate, select_regularizer,
)


def audit(root, data_root):
    root, data_root = Path(root), Path(data_root)
    summary = json.loads((root / "summary.json").read_text())
    training = json.loads((root / "training.json").read_text())
    manifest = json.loads((root / "training_manifest.json").read_text())
    checks = {
        "protocol": summary["protocol"] == manifest["protocol"] == "qwen_lowrank_geometry_v1",
        "data_hash": manifest["data_manifest_sha256"] == sha(data_root / "manifest.json"),
        "seeds": tuple(manifest["seeds"]) == SEEDS,
        "variants": tuple(manifest["registered_variants"]) == VARIANT_ORDER,
        "regularizers": tuple(manifest["regularizers"]) == REGULARIZERS,
        "loads": tuple(manifest["loads"]) == LOADS,
        "split_lock": manifest["fit_splits"] == ["train", "dev"]
                      and manifest["selection_split"] == "train"
                      and manifest["gate_split"] == "dev"
                      and manifest["test_loaded"] is False,
        "baseline_bridge": training["baseline_bridge"] is not None
                           and training["baseline_bridge"]["exact"],
        "transform_hash": training["transforms_sha256"] == sha(root / "transforms.pt"),
    }
    selected = {str(dim): select_regularizer(training["train_grid"], dim) for dim in (128, 256)}
    checks["regularizer_selection"] = selected == summary["selected_regularizers"] == training["selected_regularizers"]
    rebuilt = {}
    for label in (*VARIANT_ORDER, "random128"):
        seeds = {}
        for seed in SEEDS:
            path = root / f"dev_{label}_seed{seed}.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines() if line]
            checks[f"raw_hash_{label}_{seed}"] = training["raw_rows_sha256"][path.name] == sha(path)
            checks[f"identity_{label}_{seed}"] = all(
                row["variant"] == label and row["seed"] == seed for row in rows)
            seeds[str(seed)] = aggregate_readability(rows)
        rebuilt[label] = mean_seed_summary(seeds)
    checks["dev_recomputed"] = rebuilt == summary["dev"]
    gate = readability_gate(rebuilt, training["load_sweeps"])
    checks["gate_recomputed"] = gate == summary["readability_gate"]
    checks["selection_recomputed"] = gate["selected_variant"] == summary["selected_variant"]
    checks["test_lock"] = summary["test"] is None and summary["test_scored"] is False \
                          and not list(root.glob("test_*.jsonl"))
    result = {
        "protocol": "qwen_lowrank_geometry_v1", "audit_passed": all(checks.values()),
        "checks": checks, "stage_passed": summary["stage_passed"],
        "selected_variant": summary["selected_variant"], "test_scored": False,
        "summary_sha256": sha(root / "summary.json"),
        "training_sha256": sha(root / "training.json"),
        "manifest_sha256": sha(root / "training_manifest.json"),
        "data_manifest_sha256": sha(data_root / "manifest.json"),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(args.root, args.data_root)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["audit_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
