"""Independent aggregation audit for contextual Qwen memory geometry."""

import argparse
import json
from pathlib import Path

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_qwen_lowrank_forgetting import SEEDS
from tasks.diagnose_cbf_qwen_lowrank_geometry import (
    REGULARIZERS, aggregate_readability, mean_seed_summary,
)
from tasks.diagnose_cbf_qwen_contextual_geometry import (
    DEV_VARIANTS, POOLINGS, UNKNOWN_GRID, contextual_gate, select_context_config,
)


def audit(root, data_root, geometry_root):
    root, data_root, geometry_root = Path(root), Path(data_root), Path(geometry_root)
    summary = json.loads((root / "summary.json").read_text())
    training = json.loads((root / "training.json").read_text())
    manifest = json.loads((root / "training_manifest.json").read_text())
    source_audit = json.loads((geometry_root / "execution_audit.json").read_text())
    checks = {
        "protocol": summary["protocol"] == manifest["protocol"] == "qwen_contextual_geometry_v1",
        "data_hash": manifest["data_manifest_sha256"] == sha(data_root / "manifest.json"),
        "source_audited": source_audit["audit_passed"] and not source_audit["stage_passed"],
        "source_hashes": training["source_geometry_summary_sha256"] == sha(geometry_root / "formal/summary.json")
                         and training["source_geometry_training_sha256"] == sha(geometry_root / "formal/training.json"),
        "mean_bridge": training["mean_embedding_bridge_exact"] is True,
        "seeds": tuple(manifest["seeds"]) == SEEDS,
        "poolings": tuple(manifest["poolings"]) == POOLINGS,
        "regularizers": tuple(manifest["regularizers"]) == REGULARIZERS,
        "unknown_grid": tuple(manifest["unknown_grid"]) == UNKNOWN_GRID,
        "dev_variants": tuple(manifest["dev_variants"]) == DEV_VARIANTS,
        "split_lock": manifest["fit_splits"] == ["train", "dev"]
                      and manifest["selection_split"] == "train"
                      and manifest["gate_split"] == "dev"
                      and manifest["test_loaded"] is False,
        "transform_hash": training["transforms_sha256"] == sha(root / "contextual_transforms.pt"),
    }
    selected = {pooling: select_context_config(training["train_grid"], pooling)
                for pooling in POOLINGS}
    checks["config_selection"] = selected == training["selected_configs"] == summary["selected_configs"]
    rebuilt = {}
    for label in DEV_VARIANTS:
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
    gate = contextual_gate(rebuilt, training["load_sweeps"])
    checks["gate_recomputed"] = gate == summary["contextual_gate"]
    checks["selection_recomputed"] = gate["selected_variant"] == summary["selected_variant"]
    checks["test_lock"] = summary["test"] is None and summary["test_scored"] is False \
                          and not list(root.glob("test_*.jsonl"))
    result = {
        "protocol": "qwen_contextual_geometry_v1", "audit_passed": all(checks.values()),
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
    parser.add_argument("--geometry-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(args.root, args.data_root, args.geometry_root)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["audit_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
