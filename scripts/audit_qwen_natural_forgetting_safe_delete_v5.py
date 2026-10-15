#!/usr/bin/env python3
"""Independently audit V5 corpus, fold isolation and checkpoint predictions."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_qwen_natural_forgetting_factor_gate_v2 import BinaryHead
from tasks.train_qwen_natural_forgetting_safe_delete_v5 import (
    PROTOCOL,
    SEEDS,
    load_corpus,
)


def audit(args):
    import torch

    root = Path(args.root)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    summary = json.loads((root / "summary.json").read_text())
    stored_corpus = [json.loads(line) for line in
                     (root / "safe_delete_corpus.jsonl").read_text().splitlines() if line]
    rebuilt, source_paths = load_corpus(args.source_root)
    features = torch.load(root / "safe_delete_features.pt", map_location="cpu",
                          weights_only=False)
    artifact = torch.load(root / "safe_delete_crossfit.pt", map_location="cpu",
                          weights_only=False)
    oof = [json.loads(line) for line in
           (root / "oof_predictions.jsonl").read_text().splitlines() if line]
    identities_match = (
        len(rebuilt) == len(stored_corpus) == len(oof)
        and [row["state_id"] for row in rebuilt]
            == [row["state_id"] for row in stored_corpus]
            == [row["state_id"] for row in oof]
    )
    corpus_exact = rebuilt == stored_corpus
    tensor_exact = (
        tuple(features["feature_names"]) == tuple(summary["feature_names"])
        and torch.equal(features["features"], torch.tensor(
            [row["features"] for row in rebuilt], dtype=torch.float32))
        and torch.equal(features["targets"], torch.tensor(
            [row["safe_delete"] for row in rebuilt], dtype=torch.float32))
    )
    max_probability_error = 0.0
    action_mismatches = 0
    session_leaks = 0
    metric_counts = Counter()
    fold_counts = {str(fold): Counter() for fold in range(5)}
    for outer in range(5):
        fold = artifact["folds"][str(outer)]
        validation = [index for index, row in enumerate(rebuilt)
                      if row["outer_fold"] == outer]
        expected_sessions = {rebuilt[index]["session_id"] for index in validation}
        session_leaks += len(set(fold["train_sessions"]) & expected_sessions)
        predictions = []
        for seed in SEEDS:
            checkpoint = fold["checkpoints"][str(seed)]
            model = BinaryHead.create(
                len(artifact["feature_names"]), fold["config"]["hidden"], seed,
            )
            model.load_state_dict(checkpoint["state"])
            model.eval()
            with torch.no_grad():
                value = model(((features["features"][validation] - checkpoint["mean"])
                               / checkpoint["std"])).squeeze(-1).sigmoid()
            predictions.append(value)
        ensemble = torch.stack(predictions).mean(0)
        for local, index in enumerate(validation):
            probability = float(ensemble[local])
            max_probability_error = max(
                max_probability_error, abs(probability - float(oof[index]["probability"])),
            )
            action = probability >= float(fold["threshold"])
            action_mismatches += int(
                action != (oof[index]["action"] == "scope_top1_a0")
            )
            target = bool(rebuilt[index]["safe_delete"])
            name = "tp" if action and target else "fp" if action else "fn" if target else "tn"
            metric_counts[name] += 1
            fold_counts[str(outer)][name] += 1
    tp, fp = metric_counts["tp"], metric_counts["fp"]
    fn, tn = metric_counts["fn"], metric_counts["tn"]
    recomputed = {
        "rows": len(rebuilt), "true_positives": tp, "false_positives": fp,
        "false_negatives": fn, "true_negatives": tn,
        "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
        "delete_rate": (tp + fp) / max(1, len(rebuilt)), "by_outer_fold": {},
    }
    for outer, values in fold_counts.items():
        ftp, ffp, ffn = values["tp"], values["fp"], values["fn"]
        recomputed["by_outer_fold"][outer] = {
            "rows": sum(values.values()), "true_positives": ftp,
            "false_positives": ffp, "false_negatives": ffn,
            "precision": ftp / max(1, ftp + ffp),
            "recall": ftp / max(1, ftp + ffn),
        }
    hashes_match = all([
        summary["hashes"]["plan"] == sha("QWEN_NATURAL_FORGETTING_SAFE_DELETE_V5_PLAN.md"),
        summary["hashes"]["corpus"] == sha(root / "safe_delete_corpus.jsonl"),
        summary["hashes"]["features"] == sha(root / "safe_delete_features.pt"),
        summary["hashes"]["checkpoint"] == sha(root / "safe_delete_crossfit.pt"),
        summary["hashes"]["oof"] == sha(root / "oof_predictions.jsonl"),
        summary["hashes"]["inner_search"] == sha(root / "inner_search.json"),
        *[summary["hashes"][f"source_{name}"] == sha(path)
          for name, path in source_paths.items()],
    ])
    privileged = {
        "event_kind", "desired_source_action", "incoming_chain_id", "incoming_version",
        "safe_delete", "origin_arms", "memory", "outer_fold", "event_step",
    }
    feature_isolation = not privileged & set(summary["feature_names"])
    checks = {
        "source_and_corpus_reconstruction_exact": corpus_exact and identities_match,
        "feature_and_target_tensors_exact": tensor_exact,
        "no_train_validation_session_overlap": session_leaks == 0,
        "checkpoint_probabilities_match": max_probability_error <= 1e-7,
        "checkpoint_actions_match": action_mismatches == 0,
        "aggregate_metrics_match": recomputed == summary["metrics"],
        "privileged_fields_excluded": feature_isolation,
        "artifact_hashes_match": hashes_match,
        "held_out_sources_remained_sealed": (
            summary["calibration_loaded"] and not summary["development_loaded"]
            and not summary["test_loaded"] and not summary["external_benchmarks_loaded"]
        ),
    }
    result = {
        "protocol": PROTOCOL, "checks": checks, "passed": all(checks.values()),
        "rows": len(rebuilt), "label_distribution": dict(Counter(
            row["safe_delete"] for row in rebuilt)),
        "max_probability_error": max_probability_error,
        "action_mismatches": action_mismatches, "session_leaks": session_leaks,
        "recomputed_metrics": recomputed,
        "summary_sha256": sha(root / "summary.json"),
    }
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args), indent=2))


if __name__ == "__main__":
    main()
