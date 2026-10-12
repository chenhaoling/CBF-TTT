"""Independent aggregation audit for contextual Qwen forgetting v2."""

import argparse
import json
from pathlib import Path

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_lowrank_forgetting import POLICIES, aggregate
from tasks.train_cbf_qwen_lowrank_forgetting import SEEDS, development_gate, multi_seed_summary


def audit(root, data_root, contextual_study_root):
    root, data_root = Path(root), Path(data_root)
    study = Path(contextual_study_root); source = study / "formal"
    summary = json.loads((root / "summary.json").read_text())
    training = json.loads((root / "training.json").read_text())
    manifest = json.loads((root / "training_manifest.json").read_text())
    source_audit = json.loads((study / "execution_audit.json").read_text())
    checks = {
        "protocol": summary["protocol"] == manifest["protocol"] == "qwen_contextual_forgetting_v2",
        "data_hash": manifest["data_manifest_sha256"] == sha(data_root / "manifest.json"),
        "source_audited": source_audit["audit_passed"] and source_audit["stage_passed"]
                           and source_audit["selected_variant"] == "context_last",
        "source_hashes": training["source_summary_sha256"] == sha(source / "summary.json")
                         and training["source_training_sha256"] == sha(source / "training.json")
                         and training["source_manifest_sha256"] == sha(source / "training_manifest.json")
                         and training["source_transforms_sha256"] == sha(source / "contextual_transforms.pt"),
        "embedding_bridge": training["train_dev_embedding_bridge_exact"] is True,
        "seeds": tuple(manifest["memory_seeds"]) == SEEDS,
        "dim": manifest["dim"] == 256,
        "config": manifest["contextual_config"] == training["contextual_config"]
                  and manifest["contextual_config"]["internal"] == "context_last_r0.01"
                  and manifest["contextual_config"]["unknown_raw"] == .75,
    }
    dev_seed = {}
    for seed in SEEDS:
        dev_seed[str(seed)] = {}
        for policy in POLICIES:
            path = root / f"dev_seed{seed}_{policy}.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines() if line]
            checks[f"dev_hash_{seed}_{policy}"] = training["raw_rows_sha256"][path.name] == sha(path)
            dev_seed[str(seed)][policy] = aggregate(rows)
    rebuilt_dev = multi_seed_summary(dev_seed)
    checks["dev_recomputed"] = rebuilt_dev == summary["dev"]
    checks["gate_recomputed"] = development_gate(rebuilt_dev) == summary["dev_gate"]
    test_exists = all((root / f"test_seed{seed}_{policy}.jsonl").exists()
                      for seed in SEEDS for policy in POLICIES)
    checks["test_lock"] = test_exists == bool(summary["dev_gate"]["passed"]) \
                          == bool(summary["test_scored"]) == bool(manifest["test_scored"])
    if test_exists:
        test_seed = {}
        for seed in SEEDS:
            test_seed[str(seed)] = {}
            for policy in POLICIES:
                path = root / f"test_seed{seed}_{policy}.jsonl"
                rows = [json.loads(line) for line in path.read_text().splitlines() if line]
                checks[f"test_hash_{seed}_{policy}"] = training["raw_rows_sha256"][path.name] == sha(path)
                test_seed[str(seed)][policy] = aggregate(rows)
        checks["test_recomputed"] = multi_seed_summary(test_seed) == summary["test"]
        checks["test_embeddings_present"] = training["test_embedding_meta"] is not None
    else:
        checks["test_embeddings_absent"] = training["test_embedding_meta"] is None
    result = {
        "protocol": "qwen_contextual_forgetting_v2", "audit_passed": all(checks.values()),
        "checks": checks, "dev_gate_passed": summary["dev_gate"]["passed"],
        "stage_passed": summary["stage_passed"], "test_scored": summary["test_scored"],
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
    parser.add_argument("--contextual-study-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(args.root, args.data_root, args.contextual_study_root)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["audit_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
