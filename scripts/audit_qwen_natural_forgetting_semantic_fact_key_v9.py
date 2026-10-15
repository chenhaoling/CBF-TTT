#!/usr/bin/env python3
"""Audit CounterFact validation or frozen ZsRE transfer for semantic V9."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_qwen_natural_forgetting_fact_key_v8 import (
    build_examples,
    oof_metrics,
    pair_metrics,
    v7_prompt_pairs,
)
from tasks.train_qwen_natural_forgetting_semantic_fact_key_v9 import (
    FEATURE_NAMES,
    PLAN,
    PROTOCOL,
    SEMANTIC_DIM,
    SEMANTIC_PROJECTION_SEED,
    SPAN_EPOCH,
    SPAN_THRESHOLD,
    SemanticFactKeyBarrier,
)


PROBABILITY_TOLERANCE = 2e-7


def audit_train(args):
    import torch

    root = Path(args.root)
    summary = json.loads((root / "summary.json").read_text())
    rebuilt, exclusions = build_examples(args.counterfact_source)
    stored = [json.loads(line) for line in
              (root / "counterfact_semantic_fact_key_pairs.jsonl").read_text().splitlines()
              if line]
    features = torch.load(root / "semantic_fact_key_features.pt",
                          map_location="cpu", weights_only=False)
    checkpoint = torch.load(root / "semantic_fact_key_barrier.pt",
                            map_location="cpu", weights_only=False)
    span = torch.load(root / "crossfit_span_heads.pt", map_location="cpu",
                      weights_only=False)
    barrier = SemanticFactKeyBarrier(checkpoint)
    validation = [index for index, row in enumerate(rebuilt) if row["fold"] == 4]
    tensor = features["arms"][barrier.arm][validation]
    probabilities = barrier.probabilities(tensor).tolist()
    observed = [json.loads(line) for line in
                (root / "validation_predictions.jsonl").read_text().splitlines() if line]
    max_error = max((abs(value - row["probability"])
                     for value, row in zip(probabilities, observed)), default=0.0)
    action_mismatches = sum(
        int((value >= barrier.threshold) != bool(row["prediction"]))
        for value, row in zip(probabilities, observed))
    validation_rows = [rebuilt[index] for index in validation]
    metrics = pair_metrics(validation_rows, probabilities, barrier.threshold)
    fold_subjects = {fold: {row["left_subject"].casefold() for row in rebuilt
                            if row["fold"] == fold} for fold in range(5)}
    disjoint = all(not fold_subjects[left] & fold_subjects[right]
                   for left in range(5) for right in range(left + 1, 5))
    hashes = summary["hashes"]
    hashes_ok = all((
        hashes["plan"] == sha(PLAN),
        hashes["counterfact_source"] == sha(args.counterfact_source),
        hashes["corpus"] == sha(root / "counterfact_semantic_fact_key_pairs.jsonl"),
        hashes["features"] == sha(root / "semantic_fact_key_features.pt"),
        hashes["checkpoint"] == sha(root / "semantic_fact_key_barrier.pt"),
        hashes["span_heads"] == sha(root / "crossfit_span_heads.pt"),
        hashes["tune_search"] == sha(root / "tune_search.json"),
        hashes["validation_predictions"] == sha(root / "validation_predictions.jsonl"),
    ))
    checks = {
        "corpus_reconstruction_exact": rebuilt == stored
            and exclusions == summary["excluded"],
        "feature_manifest_exact": features["protocol"] == PROTOCOL
            and features["feature_names"] == FEATURE_NAMES
            and all(features["arms"][arm].shape[1] == len(FEATURE_NAMES[arm])
                    for arm in FEATURE_NAMES),
        "span_checkpoint_registered": span["protocol"] == PROTOCOL
            and span["epoch"] == SPAN_EPOCH
            and span["threshold"] == SPAN_THRESHOLD
            and set(span["folds"]) == {str(value) for value in range(5)},
        "checkpoint_validation_probabilities_match":
            max_error <= PROBABILITY_TOLERANCE,
        "checkpoint_validation_actions_match": action_mismatches == 0,
        "validation_metrics_match": metrics == summary["validation"],
        "subject_folds_disjoint": disjoint,
        "artifact_hashes_match": hashes_ok,
        "frozen_semantic_registration_match":
            summary["semantic_projection_seed"] == SEMANTIC_PROJECTION_SEED
            and summary["semantic_dim"] == SEMANTIC_DIM,
        "zsre_sources_remained_unloaded": summary["counterfact_loaded"]
            and not summary["zsre_calibration_loaded"]
            and not summary["zsre_test_loaded"]
            and not summary["external_benchmarks_loaded"],
    }
    return {"protocol": PROTOCOL, "stage": "counterfact_validation_audit",
            "checks": checks, "passed": all(checks.values()),
            "pairs": len(rebuilt), "max_probability_error": max_error,
            "action_mismatches": action_mismatches,
            "recomputed_validation": metrics}


def audit_transfer(args):
    import torch

    root = Path(args.root); independent = Path(args.independent_root)
    summary = json.loads((root / "summary.json").read_text())
    checkpoint = torch.load(independent / "semantic_fact_key_barrier.pt",
                            map_location="cpu", weights_only=False)
    barrier = SemanticFactKeyBarrier(checkpoint)
    features = torch.load(root / "transfer_features.pt", map_location="cpu",
                          weights_only=False)
    probabilities = barrier.probabilities(features["features"]).tolist()
    observed = [json.loads(line) for line in
                (root / "transfer_predictions.jsonl").read_text().splitlines() if line]
    pairs = v7_prompt_pairs(args.v7_root, args.data_root)
    v7 = [json.loads(line) for line in
          (Path(args.v7_root) / "result" / "oof_predictions.jsonl")
          .read_text().splitlines() if line]
    max_error, action_mismatches = 0.0, 0
    for index, ((_, _, top), source, row) in enumerate(zip(pairs, v7, observed)):
        probability = 0.0 if top is None else probabilities[index]
        max_error = max(max_error, abs(probability - row["fact_key_probability"]))
        delete = (source["action"] == "scope_top1_a0"
                  and probability >= barrier.threshold)
        action_mismatches += int(delete != (row["action"] == "scope_top1_a0"))
    metrics = oof_metrics([row for row, _, _ in pairs], observed)
    hashes = summary["hashes"]
    hashes_ok = all((
        hashes["plan"] == sha(PLAN),
        hashes["independent_summary"] == sha(independent / "summary.json"),
        hashes["independent_audit"] == sha(independent / "execution_audit.json"),
        hashes["checkpoint"] == sha(independent / "semantic_fact_key_barrier.pt"),
        hashes["span_heads"] == sha(independent / "crossfit_span_heads.pt"),
        hashes["v7_summary"] == sha(Path(args.v7_root) / "result" / "summary.json"),
        hashes["v7_audit"] == sha(Path(args.v7_root) / "execution_audit.json"),
        hashes["v7_oof"] == sha(Path(args.v7_root) / "result" / "oof_predictions.jsonl"),
        hashes["data_manifest"] == sha(Path(args.data_root) / "manifest.json"),
        hashes["features"] == sha(root / "transfer_features.pt"),
        hashes["predictions"] == sha(root / "transfer_predictions.jsonl"),
    ))
    independent_summary = json.loads((independent / "summary.json").read_text())
    checks = {
        "complete_v7_identity_alignment": len(pairs) == len(v7) == len(observed)
            and [row[0]["state_id"] for row in pairs]
            == [row["state_id"] for row in observed],
        "feature_manifest_exact": features["protocol"] == PROTOCOL
            and features["arm"] == barrier.arm
            and tuple(features["feature_names"]) == barrier.feature_names,
        "fact_key_probabilities_match": max_error <= PROBABILITY_TOLERANCE,
        "intersection_actions_match": action_mismatches == 0,
        "aggregate_metrics_match": metrics == summary["metrics"],
        "artifact_hashes_match": hashes_ok,
        "counterfact_validation_still_passed": independent_summary["gate"][
            "counterfact_validation_passed"],
        "heldout_sources_remained_sealed": summary["zsre_calibration_loaded"]
            and not summary["zsre_test_loaded"]
            and not summary["external_benchmarks_loaded"],
    }
    return {"protocol": PROTOCOL, "stage": "frozen_transfer_audit",
            "checks": checks, "passed": all(checks.values()),
            "rows": len(observed), "max_probability_error": max_error,
            "action_mismatches": action_mismatches,
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
