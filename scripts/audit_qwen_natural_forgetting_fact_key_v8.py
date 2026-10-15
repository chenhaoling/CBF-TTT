#!/usr/bin/env python3
"""Audit independent CounterFact V8 training or frozen ZsRE transfer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_qwen_natural_forgetting_fact_key_v8 import (
    FEATURE_NAMES,
    PLAN,
    PROTOCOL,
    FactKeyBarrier,
    build_examples,
    oof_metrics,
    pair_features,
    pair_metrics,
    v7_prompt_pairs,
)


def audit_train(args):
    import torch

    root = Path(args.root)
    summary = json.loads((root / "summary.json").read_text())
    rebuilt, exclusions = build_examples(args.counterfact_source)
    stored = [json.loads(line) for line in
              (root / "counterfact_fact_key_pairs.jsonl").read_text().splitlines() if line]
    features = torch.load(root / "counterfact_fact_key_features.pt",
                          map_location="cpu", weights_only=False)
    checkpoint = torch.load(root / "fact_key_barrier.pt", map_location="cpu",
                            weights_only=False)
    validation = [row for row in rebuilt if row["fold"] == 4]
    barrier = FactKeyBarrier(checkpoint)
    probabilities = [barrier.probability(pair_features(row["left"], row["right"]))
                     for row in validation]
    predictions = [json.loads(line) for line in
                   (root / "validation_predictions.jsonl").read_text().splitlines() if line]
    max_error = max((abs(probability - row["probability"])
                     for probability, row in zip(probabilities, predictions)), default=0.0)
    action_mismatches = sum(
        int((probability >= barrier.threshold) != bool(row["prediction"]))
        for probability, row in zip(probabilities, predictions))
    recomputed = pair_metrics(validation, probabilities, barrier.threshold)
    hashes_ok = all([
        summary["hashes"]["plan"] == sha(PLAN),
        summary["hashes"]["counterfact_source"] == sha(args.counterfact_source),
        summary["hashes"]["corpus"] == sha(root / "counterfact_fact_key_pairs.jsonl"),
        summary["hashes"]["features"] == sha(root / "counterfact_fact_key_features.pt"),
        summary["hashes"]["checkpoint"] == sha(root / "fact_key_barrier.pt"),
        summary["hashes"]["tune_search"] == sha(root / "tune_search.json"),
        summary["hashes"]["validation_predictions"] == sha(
            root / "validation_predictions.jsonl"),
    ])
    subject_folds = {fold: {row["left_subject"].casefold() for row in rebuilt
                            if row["fold"] == fold} for fold in range(5)}
    disjoint = all(not subject_folds[left] & subject_folds[right]
                   for left in range(5) for right in range(left + 1, 5))
    checks = {
        "corpus_reconstruction_exact": rebuilt == stored and exclusions == summary["excluded"],
        "feature_tensor_reconstruction_exact": tuple(features["feature_names"]) ==
            tuple(FEATURE_NAMES) and torch.equal(features["features"], torch.tensor(
                [pair_features(row["left"], row["right"]) for row in rebuilt],
                dtype=torch.float32)),
        "checkpoint_validation_probabilities_match": max_error <= 1e-7,
        "checkpoint_validation_actions_match": action_mismatches == 0,
        "validation_metrics_match": recomputed == summary["validation"],
        "subject_folds_disjoint": disjoint,
        "artifact_hashes_match": hashes_ok,
        "zsre_sources_remained_unloaded": summary["counterfact_loaded"] and
            not summary["zsre_calibration_loaded"] and not summary["zsre_test_loaded"] and
            not summary["external_benchmarks_loaded"],
    }
    return {"protocol": PROTOCOL, "stage": "independent_validation_audit",
            "checks": checks, "passed": all(checks.values()), "pairs": len(rebuilt),
            "max_probability_error": max_error, "action_mismatches": action_mismatches,
            "recomputed_validation": recomputed}


def audit_transfer(args):
    import torch

    root = Path(args.root)
    summary = json.loads((root / "summary.json").read_text())
    independent = Path(args.independent_root)
    checkpoint = torch.load(independent / "fact_key_barrier.pt", map_location="cpu",
                            weights_only=False)
    barrier = FactKeyBarrier(checkpoint)
    pairs = v7_prompt_pairs(args.v7_root, args.data_root)
    observed = [json.loads(line) for line in
                (root / "transfer_predictions.jsonl").read_text().splitlines() if line]
    v7_oof = [json.loads(line) for line in
              (Path(args.v7_root) / "result" / "oof_predictions.jsonl")
              .read_text().splitlines() if line]
    max_error = 0.0
    action_mismatches = 0
    rebuilt = []
    for (source, incoming, top), v7, row in zip(pairs, v7_oof, observed):
        probability = 0.0 if top is None else barrier.probability(pair_features(incoming, top))
        max_error = max(max_error, abs(probability - row["fact_key_probability"]))
        delete = v7["action"] == "scope_top1_a0" and probability >= barrier.threshold
        action_mismatches += int(delete != (row["action"] == "scope_top1_a0"))
        rebuilt.append({**row, "fact_key_probability": probability,
                        "action": "scope_top1_a0" if delete else "retain_all"})
    metrics = oof_metrics([row for row, _, _ in pairs], observed)
    hashes_ok = all([
        summary["hashes"]["plan"] == sha(PLAN),
        summary["hashes"]["independent_summary"] == sha(independent / "summary.json"),
        summary["hashes"]["independent_audit"] == sha(independent /
                                                       "execution_audit.json"),
        summary["hashes"]["checkpoint"] == sha(independent / "fact_key_barrier.pt"),
        summary["hashes"]["v7_summary"] == sha(Path(args.v7_root) /
                                                "result" / "summary.json"),
        summary["hashes"]["v7_audit"] == sha(Path(args.v7_root) /
                                              "execution_audit.json"),
        summary["hashes"]["v7_oof"] == sha(Path(args.v7_root) /
                                            "result" / "oof_predictions.jsonl"),
        summary["hashes"]["data_manifest"] == sha(Path(args.data_root) / "manifest.json"),
        summary["hashes"]["predictions"] == sha(root / "transfer_predictions.jsonl"),
    ])
    checks = {
        "complete_v7_identity_alignment": len(pairs) == len(v7_oof) == len(observed)
            and [row[0]["state_id"] for row in pairs] == [row["state_id"] for row in observed],
        "fact_key_probabilities_match": max_error <= 1e-7,
        "intersection_actions_match": action_mismatches == 0,
        "aggregate_metrics_match": metrics == summary["metrics"],
        "artifact_hashes_match": hashes_ok,
        "independent_validation_still_passed": json.loads(
            (independent / "summary.json").read_text())["gate"][
                "independent_validation_passed"],
        "heldout_sources_remained_sealed": summary["zsre_calibration_loaded"] and
            not summary["zsre_test_loaded"] and not summary["external_benchmarks_loaded"],
    }
    return {"protocol": PROTOCOL, "stage": "frozen_transfer_audit",
            "checks": checks, "passed": all(checks.values()), "rows": len(observed),
            "max_probability_error": max_error, "action_mismatches": action_mismatches,
            "recomputed_metrics": metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "transfer"))
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--counterfact-source")
    parser.add_argument("--independent-root")
    parser.add_argument("--v7-root")
    parser.add_argument("--data-root")
    args = parser.parse_args()
    result = audit_train(args) if args.command == "train" else audit_transfer(args)
    write_json(Path(args.output), result)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
