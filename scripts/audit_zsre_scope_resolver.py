"""Independent aggregation and release audit for the ZsRE scope resolver."""

import argparse
import json
from pathlib import Path

from tasks.pack_cbf_event_warmup import sha
from tasks.evaluate_zsre_contextual_readout import SEEDS as BI_SEEDS, load_split
from tasks.train_zsre_scope_resolver import (
    ARMS, CHECKPOINT_EPOCHS, HEAD_SEEDS, PAIR_TEMPLATE, aggregate, compare,
    development_gate, split_calibration, summarize,
)


def audit(root, data_root, source_study_root):
    root, data_root, study = Path(root), Path(data_root), Path(source_study_root)
    source = study / "formal"
    summary = json.loads((root / "summary.json").read_text())
    training = json.loads((root / "training.json").read_text())
    manifest = json.loads((root / "training_manifest.json").read_text())
    source_audit = json.loads((study / "execution_audit.json").read_text())
    calibration = load_split(data_root, "calibration")
    fit, tune = split_calibration(calibration)
    checks = {
        "protocol": summary["protocol"] == manifest["protocol"] == "zsre_scope_resolver_v1",
        "data_hash": manifest["data_manifest_sha256"] == sha(data_root / "manifest.json"),
        "source_audited": source_audit["audit_passed"] and not source_audit["test_scored"],
        "source_hashes": training["source_summary_sha256"] == sha(source / "summary.json")
                         and training["source_training_sha256"] == sha(source / "training.json")
                         and training["source_transforms_sha256"] == sha(source / "transforms.pt"),
        "checkpoint_hash": training["checkpoint_sha256"] == sha(root / "scope_heads.pt"),
        "arms": tuple(manifest["arms"]) == ARMS,
        "seeds": tuple(manifest["head_seeds"]) == HEAD_SEEDS
                 and tuple(manifest["bi_seeds"]) == BI_SEEDS,
        "pair_template": manifest["pair_template"] == PAIR_TEMPLATE,
        "frozen_hyperparameters": manifest["epochs"] == 200
                                  and tuple(manifest["checkpoint_epochs"]) == CHECKPOINT_EPOCHS
                                  and manifest["learning_rate"] == .01
                                  and manifest["weight_decay"] == 1e-4,
        "calibration_split": training["train_session_ids"] == [row["id"] for row in fit]
                             and training["tune_session_ids"] == [row["id"] for row in tune],
        "calibration_disjoint": not set(training["train_session_ids"]) & set(training["tune_session_ids"]),
        "feature_hashes_present": all(training[name] for name in
                                      ("train_feature_sha256", "tune_feature_sha256", "dev_feature_sha256")),
    }
    seed_map = {"bi_cosine": BI_SEEDS, "cross_linear": HEAD_SEEDS}
    dev, dev_records = {}, {arm: {} for arm in ARMS}
    for arm in ARMS:
        per_seed = {}
        for seed in seed_map[arm]:
            path = root / f"dev_{arm}_seed{seed}.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines() if line]
            checks[f"dev_hash_{arm}_{seed}"] = training["raw_rows_sha256"][path.name] == sha(path)
            checks[f"dev_identity_{arm}_{seed}"] = all(
                row["split"] == "dev" and row["seed"] == seed and row["arm"] == arm for row in rows)
            dev_records[arm][str(seed)] = rows; per_seed[str(seed)] = aggregate(rows)
        dev[arm] = summarize(per_seed)
    recomputed_comparison = compare(dev_records)
    checks["dev_recomputed"] = dev == summary["dev"]
    checks["comparison_recomputed"] = recomputed_comparison == summary["dev_comparison"]
    checks["gate_recomputed"] = development_gate(
        dev["cross_linear"], recomputed_comparison) == summary["dev_gate"]

    test_exists = all((root / f"test_{arm}_seed{seed}.jsonl").exists()
                      for arm in ARMS for seed in seed_map[arm])
    checks["test_lock"] = test_exists == bool(summary["dev_gate"]["passed"]) \
                          == bool(summary["test_scored"]) == bool(manifest["test_scored"])
    if test_exists:
        test, test_records = {}, {arm: {} for arm in ARMS}
        for arm in ARMS:
            per_seed = {}
            for seed in seed_map[arm]:
                path = root / f"test_{arm}_seed{seed}.jsonl"
                rows = [json.loads(line) for line in path.read_text().splitlines() if line]
                checks[f"test_hash_{arm}_{seed}"] = training["raw_rows_sha256"][path.name] == sha(path)
                checks[f"test_identity_{arm}_{seed}"] = all(
                    row["split"] == "test" and row["seed"] == seed and row["arm"] == arm for row in rows)
                test_records[arm][str(seed)] = rows; per_seed[str(seed)] = aggregate(rows)
            test[arm] = summarize(per_seed)
        checks["test_recomputed"] = test == summary["test"]
        checks["test_comparison_recomputed"] = compare(test_records) == summary["test_comparison"]
        checks["test_features_present"] = bool(training["test_feature_sha256"])
    else:
        checks["test_absent"] = summary["test"] is None and summary["test_comparison"] is None \
                                and training["test_feature_sha256"] is None \
                                and training["test_embedding_hashes"] is None
    return {"protocol": "zsre_scope_resolver_v1", "audit_passed": all(checks.values()),
            "checks": checks, "dev_gate_passed": summary["dev_gate"]["passed"],
            "stage_passed": summary["stage_passed"], "test_scored": summary["test_scored"],
            "summary_sha256": sha(root / "summary.json"),
            "training_sha256": sha(root / "training.json"),
            "manifest_sha256": sha(root / "training_manifest.json"),
            "data_manifest_sha256": sha(data_root / "manifest.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True); parser.add_argument("--data-root", required=True)
    parser.add_argument("--source-study-root", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(); result = audit(args.root, args.data_root, args.source_study_root)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result, indent=2))
    if not result["audit_passed"]: raise SystemExit(1)


if __name__ == "__main__": main()
