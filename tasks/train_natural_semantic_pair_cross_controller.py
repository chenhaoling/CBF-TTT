"""Train nested utility/risk heads on frozen symmetric pair cross-features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

from tasks.build_natural_semantic_relation_pairs import load_jsonl
from tasks.evaluate_natural_semantic_embedding_pairs import (
    FOLDS, PAIR_MANIFEST_SHA256, PAIR_ROWS_SHA256,
    binary_metrics, rank_metrics,
)
from tasks.extract_natural_semantic_pair_cross_embeddings import load_cross_embeddings
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_natural_semantic_projection import EPOCHS, SEEDS
from tasks.train_natural_semantic_residual_controller import (
    controller_probabilities, decision_metric, search_decision,
    select_snapshot, train_controller,
)


PROTOCOL = "natural_semantic_pair_cross_controller_v1"
CROSS_AUDIT_SHA256 = (
    "b0ff99e28a13e8744527b86ccf9b70a20504e9b1e6ccbf403effb3fd482cfcc3"
)
ZERO_SHOT_SUMMARY_SHA256 = (
    "fa24986f598da28f24523ffd2c71f7e3a817123e8a40680362012e48942de152"
)
FEATURES = ("symmetric_cross_embedding_2560", "zero_shot_yes_no_margin",
            "character_trigram_jaccard")


def verify_release(pair_root, cross_root, zero_root):
    pair_root, cross_root, zero_root = map(Path, (pair_root, cross_root, zero_root))
    audit = json.loads((cross_root / "audit.json").read_text())
    zero = json.loads((zero_root / "results/summary.json").read_text())
    checks = (
        sha(pair_root / "manifest.json") == PAIR_MANIFEST_SHA256,
        sha(pair_root / "pairs.jsonl") == PAIR_ROWS_SHA256,
        sha(cross_root / "audit.json") == CROSS_AUDIT_SHA256,
        audit.get("audit_passed") is True,
        sha(zero_root / "results/summary.json") == ZERO_SHOT_SUMMARY_SHA256,
        zero.get("terminal_status") == "completed_natural_semantic_pair_cross_zero_shot",
        (cross_root / "runner_exit_code.txt").read_text().strip() == "0",
        (zero_root / "runner_exit_code.txt").read_text().strip() == "0",
    )
    if not all(checks):
        raise ValueError("cross controller requires frozen audited dependencies")
    return {"pair_manifest_sha256": sha(pair_root / "manifest.json"),
            "pairs_sha256": sha(pair_root / "pairs.jsonl"),
            "cross_audit_sha256": sha(cross_root / "audit.json"),
            "zero_shot_summary_sha256": sha(zero_root / "results/summary.json")}


def train(args):
    import torch
    output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
    dependencies = verify_release(args.pair_root, args.cross_root, args.zero_root)
    pairs = load_jsonl(Path(args.pair_root) / "pairs.jsonl")
    pair_ids, cross, margins = load_cross_embeddings(args.cross_root)
    if pair_ids != [row["pair_id"] for row in pairs]:
        raise ValueError("pair order mismatch")
    trigrams = torch.tensor([float(row["character_trigram_jaccard"]) for row in pairs])
    features = torch.cat((cross.float(), margins[:, None].float(), trigrams[:, None]), dim=1)
    labels = torch.tensor([float(row["same_relation"]) for row in pairs])
    torch.manual_seed(2241)
    torch.cuda.reset_peak_memory_stats()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.perf_counter()
    artifacts, selections, rows = {}, {}, []
    seed_records = {str(seed): [] for seed in SEEDS}
    fold_metrics = {}
    for outer in range(FOLDS):
        inner = (outer + 1) % FOLDS
        train_folds = sorted(set(range(FOLDS)) - {outer, inner})
        train_indices = torch.tensor([index for index, row in enumerate(pairs)
                                      if int(row["outer_fold"]) in train_folds])
        inner_indices = torch.tensor([index for index, row in enumerate(pairs)
                                      if int(row["outer_fold"]) == inner])
        outer_indices = torch.tensor([index for index, row in enumerate(pairs)
                                      if int(row["outer_fold"]) == outer])
        seed_artifacts, inner_probabilities, outer_probabilities = {}, [], []
        for seed in SEEDS:
            effective_seed = seed + outer * 1000
            snapshots, mean, std, curve = train_controller(
                features, labels, train_indices, effective_seed)
            candidates = []
            for epoch in EPOCHS:
                probability = controller_probabilities(
                    features, inner_indices, snapshots[epoch], mean, std,
                    effective_seed)
                selection, grid = search_decision(labels[inner_indices], probability)
                candidates.append({"epoch": epoch, "selection": selection,
                                   "decision_grid": grid, "probability": probability})
            selected = select_snapshot(candidates)
            state = snapshots[selected["epoch"]]
            outer_probability = controller_probabilities(
                features, outer_indices, state, mean, std, effective_seed)
            outer_metric, outer_score, outer_prediction = decision_metric(
                labels[outer_indices], outer_probability,
                selected["selection"]["threshold"],
                selected["selection"]["risk_penalty"])
            inner_probabilities.append(selected["probability"])
            outer_probabilities.append(outer_probability)
            seed_records[str(seed)].extend(zip(
                labels[outer_indices].tolist(), outer_score, outer_prediction))
            seed_artifacts[str(seed)] = {
                "seed": seed, "effective_seed": effective_seed,
                "selected_epoch": selected["epoch"],
                "selected_decision": selected["selection"],
                "outer_metrics": outer_metric,
                "candidate_selections": [{"epoch": value["epoch"],
                                           "selection": value["selection"],
                                           "decision_grid": value["decision_grid"]}
                                          for value in candidates],
                "training_curve": curve, "mean": mean, "std": std, "state": state,
            }
        inner_ensemble = torch.stack(inner_probabilities).mean(dim=0)
        outer_ensemble = torch.stack(outer_probabilities).mean(dim=0)
        ensemble_selection, ensemble_grid = search_decision(
            labels[inner_indices], inner_ensemble)
        ensemble_metric, ensemble_score, ensemble_prediction = decision_metric(
            labels[outer_indices], outer_ensemble,
            ensemble_selection["threshold"], ensemble_selection["risk_penalty"])
        fold_metrics[str(outer)] = ensemble_metric
        selections[str(outer)] = {
            "outer_fold": outer, "inner_fold": inner, "train_folds": train_folds,
            "ensemble_decision": ensemble_selection,
            "ensemble_decision_grid": ensemble_grid,
            "outer_metrics": ensemble_metric,
            "seeds": {seed: {key: value for key, value in artifact.items()
                             if key not in ("state", "mean", "std")}
                      for seed, artifact in seed_artifacts.items()},
        }
        artifacts[str(outer)] = {"inner_fold": inner, "train_folds": train_folds,
                                 "ensemble_decision": ensemble_selection,
                                 "seeds": seed_artifacts}
        for local, pair_index in enumerate(outer_indices.tolist()):
            pair = pairs[pair_index]
            rows.append({
                "pair_id": pair["pair_id"], "outer_fold": outer,
                "kind": pair["kind"], "same_relation": int(pair["same_relation"]),
                "utility_probability": float(outer_ensemble[local, 0]),
                "risk_probability": float(outer_ensemble[local, 1]),
                "decision_score": ensemble_score[local],
                "prediction": ensemble_prediction[local],
            })
    rows.sort(key=lambda row: row["pair_id"])
    labels_all = [row["same_relation"] for row in rows]
    ensemble = binary_metrics(labels_all, [row["prediction"] for row in rows])
    ensemble.update(rank_metrics(labels_all, [row["decision_score"] for row in rows]))
    per_seed = {}
    for seed, values in seed_records.items():
        metric = binary_metrics([row[0] for row in values], [row[2] for row in values])
        metric.update(rank_metrics([row[0] for row in values], [row[1] for row in values]))
        per_seed[seed] = metric
    by_kind = {kind: binary_metrics(
        [row["same_relation"] for row in rows if row["kind"] == kind],
        [row["prediction"] for row in rows if row["kind"] == kind])
        for kind in sorted({row["kind"] for row in rows})}
    checks = {
        "complete_nested_oof_coverage": (
            len(rows) == len(pairs) and len({row["pair_id"] for row in rows}) == len(pairs)),
        "all_inner_seed_and_ensemble_constraints_feasible": all(
            value["ensemble_decision"]["constraints_feasible"]
            and all(seed["selected_decision"]["constraints_feasible"]
                    for seed in value["seeds"].values())
            for value in selections.values()),
        "ensemble_balanced_accuracy_at_least_point_80":
            ensemble["balanced_accuracy"] >= .80,
        "ensemble_auroc_at_least_point_85": ensemble["auroc"] >= .85,
        "all_fold_balanced_accuracy_at_least_point_70": all(
            value["balanced_accuracy"] >= .70 for value in fold_metrics.values()),
        "all_seed_balanced_accuracy_at_least_point_75": all(
            value["balanced_accuracy"] >= .75 for value in per_seed.values()),
        "lexical_hard_negative_specificity_at_least_point_80":
            by_kind["lexical_hard_negative"]["negative_specificity"] >= .80,
    }
    rows_path = output / "oof_cross_controller.jsonl"
    with rows_path.open("x") as sink:
        for row in rows:
            sink.write(json.dumps(row) + "\n")
    checkpoint_path = output / "cross_controllers.pt"
    torch.save({"protocol": PROTOCOL, "feature_spec": FEATURES,
                "feature_dim": features.shape[1], "seeds": SEEDS,
                "folds": artifacts}, checkpoint_path)
    result = {
        "protocol": PROTOCOL, "rows": len(rows), "feature_spec": list(FEATURES),
        "feature_dim": features.shape[1], "ensemble_oof": ensemble,
        "per_seed_oof": per_seed, "per_fold_ensemble": fold_metrics,
        "by_kind": by_kind, "selection": selections,
        "controller_gate": {"checks": checks, "passed": all(checks.values())},
        "dependencies": dependencies,
        "files_sha256": {rows_path.name: sha(rows_path),
                         checkpoint_path.name: sha(checkpoint_path)},
        "resources": {"seconds": time.perf_counter() - started,
                      "gpu": torch.cuda.get_device_name(),
                      "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                      "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30},
        "relation_ids_used_as_training_labels_only": True,
        "model_outcomes_loaded": False, "sealed_benchmarks_loaded": False,
        "next_stage": ("apply_cross_controller_to_natural_memory_episodes"
                       if all(checks.values())
                       else "stop_learned_semantic_controller_iteration_on_observed_sources"),
        "terminal_status": "completed_natural_semantic_pair_cross_controller",
    }
    write_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-root", required=True)
    parser.add_argument("--cross-root", required=True)
    parser.add_argument("--zero-root", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(train(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
