"""Independent aggregation audit for ZsRE contrastive key alignment."""

import argparse
import json
from pathlib import Path

from tasks.pack_cbf_event_warmup import sha
from tasks.evaluate_zsre_contextual_readout import SEEDS, aggregate, gate, summarize
from tasks.train_zsre_contrastive_keys import ARMS, comparison


def audit(root, data_root, source_study_root):
    root, data_root, study = Path(root), Path(data_root), Path(source_study_root)
    source = study / "formal"
    summary = json.loads((root / "summary.json").read_text())
    training = json.loads((root / "training.json").read_text())
    manifest = json.loads((root / "training_manifest.json").read_text())
    source_audit = json.loads((study / "execution_audit.json").read_text())
    checks = {"protocol": summary["protocol"] == manifest["protocol"] == "zsre_contrastive_keys_v1",
              "data_hash": manifest["data_manifest_sha256"] == sha(data_root / "manifest.json"),
              "source_audited": source_audit["audit_passed"] and not source_audit["dev_gate_passed"]
                                 and not source_audit["test_scored"],
              "source_hashes": training["source_summary_sha256"] == sha(source / "summary.json")
                               and training["source_training_sha256"] == sha(source / "training.json")
                               and training["source_transforms_sha256"] == sha(source / "transforms.pt"),
              "identity_bridge": training["identity_bridge_exact"],
              "checkpoint_hash": training["checkpoint_sha256"] == sha(root / "alignment_heads.pt"),
              "arms": tuple(manifest["arms"]) == ARMS,
              "seeds": tuple(manifest["seeds"]) == SEEDS,
              "frozen_hyperparameters": manifest["rank"] == 16 and manifest["epochs"] == 300
                                        and manifest["temperature"] == .07
                                        and manifest["locality_margin"] == .2
                                        and manifest["locality_weight"] == .5}
    dev, dev_records = {}, {arm: {} for arm in ARMS}
    for arm in ARMS:
        per_seed = {}
        for seed in SEEDS:
            path = root / f"dev_{arm}_seed{seed}.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines() if line]
            checks[f"dev_hash_{arm}_{seed}"] = training["raw_rows_sha256"][path.name] == sha(path)
            checks[f"dev_identity_{arm}_{seed}"] = all(
                row["split"] == "dev" and row["seed"] == seed for row in rows)
            dev_records[arm][str(seed)] = rows; per_seed[str(seed)] = aggregate(rows)
        dev[arm] = summarize(per_seed)
    checks["dev_recomputed"] = dev == summary["dev"]
    checks["gate_recomputed"] = gate(dev["relation_aware"]) == summary["dev_gate"]
    test_exists = all((root / f"test_{arm}_seed{seed}.jsonl").exists()
                      for arm in ARMS for seed in SEEDS)
    checks["test_lock"] = test_exists == bool(summary["dev_gate"]["passed"]) \
                          == bool(summary["test_scored"]) == bool(manifest["test_scored"])
    if test_exists:
        test, test_records = {}, {arm: {} for arm in ARMS}
        for arm in ARMS:
            per_seed = {}
            for seed in SEEDS:
                path = root / f"test_{arm}_seed{seed}.jsonl"
                rows = [json.loads(line) for line in path.read_text().splitlines() if line]
                checks[f"test_hash_{arm}_{seed}"] = training["raw_rows_sha256"][path.name] == sha(path)
                checks[f"test_identity_{arm}_{seed}"] = all(
                    row["split"] == "test" and row["seed"] == seed for row in rows)
                test_records[arm][str(seed)] = rows; per_seed[str(seed)] = aggregate(rows)
            test[arm] = summarize(per_seed)
        checks["test_recomputed"] = test == summary["test"]
        checks["comparison_recomputed"] = comparison(test_records) == summary["test_comparison"]
        checks["test_embeddings_present"] = training["test_embedding_hashes"] is not None
    else:
        checks["test_absent"] = summary["test"] is None and summary["test_comparison"] is None \
                                and training["test_embedding_hashes"] is None
    result = {"protocol": "zsre_contrastive_keys_v1", "audit_passed": all(checks.values()),
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
    parser.add_argument("--source-study-root", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(); result = audit(args.root, args.data_root, args.source_study_root)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result, indent=2))
    if not result["audit_passed"]: raise SystemExit(1)


if __name__ == "__main__": main()
