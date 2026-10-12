"""Independent audit for gated ZsRE forgetting development/test evaluation."""

import argparse
import json
from pathlib import Path

from tasks.build_zsre_counterfactual_forgetting import PROTOCOL as DATA_PROTOCOL
from tasks.pack_cbf_event_warmup import sha
from tasks.train_zsre_decomposed_scope_gate import GATE_SEEDS
from tasks.evaluate_zsre_hard_retain_forgetting import aggregate
from tasks.train_zsre_forgetting_controller import CONTROLLER_SEEDS, paired_comparison, summarize_seed_metrics
from tasks.train_zsre_counterfactual_forgetting_controller_v3 import PROTOCOL as SOURCE_PROTOCOL
from tasks.evaluate_zsre_counterfactual_forgetting_controller import PROTOCOL, development_gate


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def recompute(root, split, training, comparison_seed):
    controller_records, controller_metrics = {}, {}
    checks = {}
    for seed in CONTROLLER_SEEDS:
        path = root / f"{split}_controller_seed{seed}.jsonl"; rows = read_rows(path)
        controller_records[str(seed)] = rows; controller_metrics[str(seed)] = aggregate(rows)
        checks[f"{split}_controller_hash_{seed}"] = training["raw_rows_sha256"][path.name] == sha(path)
        checks[f"{split}_controller_identity_{seed}"] = all(
            row["split"] == split and row["seed"] == seed for row in rows)
    baseline_records, baseline_metrics = {}, {}
    for seed in GATE_SEEDS:
        path = root / f"{split}_cosine_local_seed{seed}.jsonl"; rows = read_rows(path)
        baseline_records[str(seed)] = rows; baseline_metrics[str(seed)] = aggregate(rows)
        checks[f"{split}_baseline_hash_{seed}"] = training["raw_rows_sha256"][path.name] == sha(path)
        checks[f"{split}_baseline_identity_{seed}"] = all(
            row["split"] == split and row["seed"] == seed for row in rows)
    result = {"controller": summarize_seed_metrics(controller_metrics),
              "cosine_local": summarize_seed_metrics(baseline_metrics),
              "comparison": paired_comparison(controller_records, baseline_records, comparison_seed)}
    return result, checks


def audit(root, data_root, controller_root, source_root, scope_root):
    root, data_root, controller_root, source, scope = map(
        Path, (root, data_root, controller_root, source_root, scope_root))
    summary = json.loads((root / "summary.json").read_text())
    training = json.loads((root / "training.json").read_text())
    manifest = json.loads((root / "training_manifest.json").read_text())
    checks = {
        "protocol": summary["protocol"] == manifest["protocol"] == PROTOCOL,
        "source_protocol": manifest["source_protocol"] == SOURCE_PROTOCOL,
        "data_protocol": manifest["data_protocol"] == DATA_PROTOCOL,
        "data_hash": manifest["data_manifest_sha256"] == sha(data_root / "manifest.json"),
        "controller_hashes": training["controller_summary_sha256"] == sha(controller_root / "summary.json")
                             and training["controller_training_sha256"] == sha(controller_root / "training.json")
                             and training["controller_manifest_sha256"] == sha(
                                 controller_root / "training_manifest.json")
                             and training["controller_checkpoint_sha256"] == sha(
                                 controller_root / "counterfactual_forgetting_action_oof_gate.pt"),
        "source_hashes": training["source_summary_sha256"] == sha(source / "summary.json")
                         and training["source_training_sha256"] == sha(source / "training.json")
                         and training["source_transforms_sha256"] == sha(source / "transforms.pt"),
        "scope_hashes": training["scope_training_sha256"] == sha(scope / "training.json")
                        and training["scope_checkpoint_sha256"] == sha(scope / "scope_gates.pt"),
        "frozen_controller": manifest["controller_threshold"] == .4
                             and manifest["controller_epochs"] == 200,
    }
    dev, dev_checks = recompute(root, "dev", training, 9301); checks.update(dev_checks)
    checks["dev_recomputed"] = dev == summary["dev"]
    gate = development_gate(dev["controller"], dev["comparison"])
    checks["dev_gate_recomputed"] = gate == summary["development_gate"]
    checks["dev_units"] = dev["comparison"]["paired_subject_units"] == 256
    test_exists = all((root / f"test_controller_seed{seed}.jsonl").exists()
                      for seed in CONTROLLER_SEEDS) and all(
                          (root / f"test_cosine_local_seed{seed}.jsonl").exists() for seed in GATE_SEEDS)
    checks["release_consistency"] = summary["test_scored"] == manifest["test_loaded"] \
        == manifest["test_scored"] == gate["passed"] == test_exists
    if test_exists:
        test, test_checks = recompute(root, "test", training, 9401); checks.update(test_checks)
        checks["test_recomputed"] = test == summary["test"]
        checks["test_units"] = test["comparison"]["paired_subject_units"] == 516
    else:
        checks["test_absent"] = summary["test"] is None and not list(root.glob("test_*.jsonl"))
    return {"protocol": PROTOCOL, "audit_passed": all(checks.values()), "checks": checks,
            "dev_loaded": True, "test_loaded": test_exists, "test_scored": test_exists,
            "summary_sha256": sha(root / "summary.json"),
            "training_sha256": sha(root / "training.json"),
            "manifest_sha256": sha(root / "training_manifest.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True); parser.add_argument("--data-root", required=True)
    parser.add_argument("--controller-root", required=True); parser.add_argument("--source-root", required=True)
    parser.add_argument("--scope-root", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(); result = audit(
        args.root, args.data_root, args.controller_root, args.source_root, args.scope_root)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result, indent=2))
    if not result["audit_passed"]: raise SystemExit(1)


if __name__ == "__main__": main()
