"""Train and calibrate the preregistered FewRel selective forgetting controller."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import random
from pathlib import Path
import time

from tasks.build_fewrel_selective_forgetting import load_jsonl
from tasks.evaluate_natural_semantic_embedding_pairs import (
    binary_metrics, rank_metrics, select_threshold,
)
from tasks.extract_fewrel_selective_pair_embeddings import (
    PAIR_AUDIT_SHA256, PAIR_MANIFEST_SHA256, development_pairs, load_embeddings,
)
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_natural_semantic_projection import EPOCHS
from tasks.train_natural_semantic_residual_controller import (
    controller_probabilities, train_controller,
)


PROTOCOL = "fewrel_selective_controller_development_v1"
EMBEDDING_AUDIT_SHA256 = (
    "9fd6a12b5a1c28f676aa1febb44af30c9dcd654c6b3941afaaf1f2c79a396857"
)
SEEDS = (23031, 23037, 23043)
RISK_PENALTIES = (0.0, .5, 1.0, 2.0)
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 2341
FEATURES = ("symmetric_cross_embedding_2560", "zero_shot_yes_no_margin",
            "character_trigram_jaccard")


def metric(labels, scores, threshold):
    predictions = [int(score >= threshold) for score in scores]
    result = binary_metrics(labels, predictions)
    result.update(rank_metrics(labels, scores))
    result["false_deletion_rate"] = 1 - result["negative_specificity"]
    return result, predictions


def full_binary_selection(labels, probabilities):
    candidates = []
    for penalty in RISK_PENALTIES:
        scores = (probabilities[:, 0] - penalty * probabilities[:, 1]).tolist()
        threshold, values, feasible = select_threshold(labels, scores)
        values.update(rank_metrics(labels, scores))
        values["false_deletion_rate"] = 1 - values["negative_specificity"]
        candidates.append({"risk_penalty": penalty, "threshold": threshold,
                           "constraints_feasible": feasible, "metrics": values})
    selected = max(candidates, key=lambda value: (
        value["constraints_feasible"], value["metrics"]["balanced_accuracy"],
        value["metrics"]["positive_recall"],
        value["metrics"]["negative_specificity"], -value["risk_penalty"]))
    return selected, candidates


def score_boundaries(labels, scores):
    """Return exact >= threshold confusion counts after every tied score group."""
    positive, negative = sum(labels), len(labels) - sum(labels)
    groups = []
    for score, label in sorted(zip(scores, labels), reverse=True):
        if not groups or score != groups[-1][0]:
            groups.append([score, 0, 0])
        groups[-1][1 + int(label)] += 1
    tp = fp = 0
    output = []
    for score, group_negative, group_positive in groups:
        tp += group_positive; fp += group_negative
        fn, tn = positive - tp, negative - fp
        precision = tp / (tp + fp) if tp + fp else 0.0
        output.append({
            "threshold": score, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "positive_recall": tp / positive, "negative_specificity": tn / negative,
            "deletion_precision": precision, "false_deletion_rate": fp / negative,
        })
    return output


def relation_bootstrap_draws(relations, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    unique = sorted(set(relations))
    generator = random.Random(seed)
    return unique, [[generator.randrange(len(unique)) for _ in unique] for _ in range(draws)]


def cluster_false_deletion_upper(labels, predictions, relations, draw_spec):
    unique, draws = draw_spec
    index = {relation: position for position, relation in enumerate(unique)}
    fp = [0] * len(unique); negative = [0] * len(unique)
    for label, prediction, relation in zip(labels, predictions, relations):
        if not label:
            position = index[relation]
            negative[position] += 1
            fp[position] += int(prediction == 1)
    values = []
    for draw in draws:
        denominator = sum(negative[position] for position in draw)
        numerator = sum(fp[position] for position in draw)
        values.append(numerator / denominator if denominator else 0.0)
    values.sort()
    return values[min(len(values) - 1, math.ceil(.975 * len(values)) - 1)]


def retain_selection(labels, scores):
    positive, negative = sum(labels), len(labels) - sum(labels)
    groups = []
    for score, label in sorted(zip(scores, labels)):
        if not groups or score != groups[-1][0]:
            groups.append([score, 0, 0])
        groups[-1][1 + int(label)] += 1
    candidates, true_retain, false_retain = [], 0, 0
    for threshold, group_negative, group_positive in groups:
        true_retain += group_negative; false_retain += group_positive
        retained = true_retain + false_retain
        precision = true_retain / retained if retained else 0.0
        candidates.append({"threshold": threshold, "retain_precision": precision,
                           "negative_recall": true_retain / negative,
                           "retained": retained, "false_retain": false_retain,
                           "positive": positive, "negative": negative})
    feasible = [value for value in candidates
                if value["retain_precision"] >= .90 and value["negative_recall"] >= .25]
    pool = feasible or candidates
    selected = max(pool, key=lambda value: (
        value["retain_precision"] >= .90 and value["negative_recall"] >= .25,
        value["negative_recall"], value["retain_precision"], -value["threshold"]))
    return selected, bool(feasible)


def selective_selection(labels, probabilities, relations, draw_spec):
    candidates = []
    for penalty in RISK_PENALTIES:
        scores = (probabilities[:, 0] - penalty * probabilities[:, 1]).tolist()
        boundaries = score_boundaries(labels, scores)
        empirical = [value for value in boundaries
                     if value["deletion_precision"] >= .90
                     and value["positive_recall"] >= .25]
        empirical.sort(key=lambda value: (-value["positive_recall"],
                                          value["false_deletion_rate"],
                                          -value["deletion_precision"]))
        chosen = None
        for value in empirical:
            predictions = [int(score >= value["threshold"]) for score in scores]
            upper = cluster_false_deletion_upper(
                labels, predictions, relations, draw_spec)
            if upper <= .05:
                chosen = {**value, "false_deletion_relation_bootstrap_upper_95": upper}
                break
        if chosen is None:
            diagnostic_pool = empirical or boundaries
            diagnostic = min(diagnostic_pool, key=lambda value: (
                value["false_deletion_rate"], -value["deletion_precision"],
                -value["positive_recall"]))
            predictions = [int(score >= diagnostic["threshold"]) for score in scores]
            chosen = {**diagnostic,
                      "false_deletion_relation_bootstrap_upper_95":
                          cluster_false_deletion_upper(
                              labels, predictions, relations, draw_spec)}
        retain, retain_feasible = retain_selection(labels, scores)
        if retain["threshold"] >= chosen["threshold"]:
            retain = {**retain, "threshold": math.nextafter(
                chosen["threshold"], -math.inf)}
        actions = ["delete" if score >= chosen["threshold"] else
                   "retain" if score <= retain["threshold"] else "defer"
                   for score in scores]
        coverage = sum(action != "defer" for action in actions) / len(actions)
        gate = {
            "false_deletion_relation_bootstrap_upper_at_most_point_05":
                chosen["false_deletion_relation_bootstrap_upper_95"] <= .05,
            "deletion_precision_at_least_point_90": chosen["deletion_precision"] >= .90,
            "positive_recall_at_least_point_25": chosen["positive_recall"] >= .25,
            "non_deferred_coverage_at_least_point_20": coverage >= .20,
        }
        candidates.append({
            "risk_penalty": penalty, "delete": chosen, "retain": retain,
            "retain_constraints_feasible": retain_feasible,
            "non_deferred_coverage": coverage,
            "action_counts": {action: actions.count(action)
                              for action in ("delete", "retain", "defer")},
            "gate": {"checks": gate, "passed": all(gate.values())},
        })
    selected = max(candidates, key=lambda value: (
        value["gate"]["passed"], value["delete"]["positive_recall"],
        -value["delete"]["false_deletion_relation_bootstrap_upper_95"],
        value["delete"]["deletion_precision"], value["non_deferred_coverage"],
        -value["risk_penalty"]))
    return selected, candidates


def apply_decisions(probabilities, full, selective):
    full_scores = (probabilities[:, 0]
                   - full["risk_penalty"] * probabilities[:, 1]).tolist()
    full_predictions = [int(score >= full["threshold"]) for score in full_scores]
    selective_scores = (probabilities[:, 0]
                         - selective["risk_penalty"] * probabilities[:, 1]).tolist()
    actions = ["delete" if score >= selective["delete"]["threshold"] else
               "retain" if score <= selective["retain"]["threshold"] else "defer"
               for score in selective_scores]
    return full_scores, full_predictions, selective_scores, actions


def verify_release(pair_root, embedding_root):
    pair_root, embedding_root = Path(pair_root), Path(embedding_root)
    audit = json.loads((embedding_root / "audit.json").read_text())
    if any((
        sha(pair_root / "data/manifest.json") != PAIR_MANIFEST_SHA256,
        sha(pair_root / "audit.json") != PAIR_AUDIT_SHA256,
        sha(embedding_root / "audit.json") != EMBEDDING_AUDIT_SHA256,
        audit.get("audit_passed") is not True,
        audit.get("internal_test_loaded") is not False,
        audit.get("external_test_loaded") is not False,
        (embedding_root / "runner_exit_code.txt").read_text().strip() != "0",
    )):
        raise ValueError("FewRel controller requires frozen audited development features")
    return {
        "pair_manifest_sha256": PAIR_MANIFEST_SHA256,
        "pair_audit_sha256": PAIR_AUDIT_SHA256,
        "embedding_audit_sha256": EMBEDDING_AUDIT_SHA256,
    }


def baseline(train_rows, dev_rows, field):
    train_labels = [int(row["same_relation"]) for row in train_rows]
    train_scores = [float(row[field]) for row in train_rows]
    threshold, training, feasible = select_threshold(train_labels, train_scores)
    dev_labels = [int(row["same_relation"]) for row in dev_rows]
    dev_scores = [float(row[field]) for row in dev_rows]
    values, _ = metric(dev_labels, dev_scores, threshold)
    return {"field": field, "threshold_selected_on_train": threshold,
            "train_constraints_feasible": feasible, "train_metrics": training,
            "development_metrics": values}


def train(args):
    import torch

    output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
    dependencies = verify_release(args.pair_root, args.embedding_root)
    all_pairs = load_jsonl(Path(args.pair_root) / "data/pairs.jsonl")
    pairs = development_pairs(all_pairs)
    pair_ids, partitions, cross, margins = load_embeddings(args.embedding_root)
    if pair_ids != [row["pair_id"] for row in pairs] \
            or partitions != [row["partition"] for row in pairs]:
        raise ValueError("FewRel pair feature order mismatch")
    trigrams = torch.tensor([float(row["character_trigram_jaccard"]) for row in pairs])
    features = torch.cat((cross.float(), margins[:, None].float(), trigrams[:, None]), dim=1)
    labels = torch.tensor([float(row["same_relation"]) for row in pairs])
    train_indices = torch.tensor([index for index, row in enumerate(pairs)
                                  if row["partition"] == "train"])
    dev_indices = torch.tensor([index for index, row in enumerate(pairs)
                                if row["partition"] == "development"])
    dev_rows = [pairs[index] for index in dev_indices.tolist()]
    dev_labels = [int(row["same_relation"]) for row in dev_rows]
    dev_relations = [row["left_relation_id"] for row in dev_rows]
    draw_spec = relation_bootstrap_draws(dev_relations)

    torch.manual_seed(2301)
    torch.cuda.reset_peak_memory_stats()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.perf_counter()
    seed_artifacts, seed_summaries, selected_probabilities = {}, {}, []
    for seed in SEEDS:
        snapshots, mean, std, curve = train_controller(
            features, labels, train_indices, seed)
        epochs = []
        for epoch in EPOCHS:
            probability = controller_probabilities(
                features, dev_indices, snapshots[epoch], mean, std, seed)
            score = (probability[:, 0] - probability[:, 1]).tolist()
            ranking = rank_metrics(dev_labels, score)
            epochs.append({"epoch": epoch, "rank_metrics": ranking,
                           "probability": probability})
        selected_epoch = max(epochs, key=lambda value: (
            value["rank_metrics"]["auroc"], value["rank_metrics"]["average_precision"],
            -value["epoch"]))
        probability = selected_epoch["probability"]
        full, full_grid = full_binary_selection(dev_labels, probability)
        selective, selective_grid = selective_selection(
            dev_labels, probability, dev_relations, draw_spec)
        selected_probabilities.append(probability)
        seed_summaries[str(seed)] = {
            "seed": seed, "selected_epoch": selected_epoch["epoch"],
            "epoch_rank_metrics": [{"epoch": value["epoch"],
                                    "rank_metrics": value["rank_metrics"]}
                                   for value in epochs],
            "full_binary": full, "full_binary_grid": full_grid,
            "selective": selective, "selective_grid": selective_grid,
            "training_curve": curve,
        }
        seed_artifacts[str(seed)] = {
            "seed": seed, "selected_epoch": selected_epoch["epoch"],
            "state": snapshots[selected_epoch["epoch"]], "mean": mean, "std": std,
            "full_binary": full, "selective": selective,
        }

    ensemble_probability = torch.stack(selected_probabilities).mean(dim=0)
    ensemble_full, ensemble_full_grid = full_binary_selection(
        dev_labels, ensemble_probability)
    ensemble_selective, ensemble_selective_grid = selective_selection(
        dev_labels, ensemble_probability, dev_relations, draw_spec)
    full_scores, full_predictions, selective_scores, actions = apply_decisions(
        ensemble_probability, ensemble_full, ensemble_selective)
    development_rows = []
    for local, pair_index in enumerate(dev_indices.tolist()):
        pair = pairs[pair_index]
        development_rows.append({
            "pair_id": pair["pair_id"], "partition": pair["partition"],
            "kind": pair["kind"], "left_relation_id": pair["left_relation_id"],
            "right_relation_id": pair["right_relation_id"],
            "same_relation": int(pair["same_relation"]),
            "utility_probability": float(ensemble_probability[local, 0]),
            "risk_probability": float(ensemble_probability[local, 1]),
            "full_score": full_scores[local], "full_prediction": full_predictions[local],
            "selective_score": selective_scores[local], "selective_action": actions[local],
        })
    development_rows.sort(key=lambda row: row["pair_id"])
    predictions_path = output / "development_predictions.jsonl"
    with predictions_path.open("x") as sink:
        for row in development_rows:
            sink.write(json.dumps(row) + "\n")
    checkpoint_path = output / "fewrel_selective_controllers.pt"
    torch.save({"protocol": PROTOCOL, "feature_spec": FEATURES,
                "feature_dim": features.shape[1], "seeds": SEEDS,
                "controllers": seed_artifacts,
                "ensemble_full_binary": ensemble_full,
                "ensemble_selective": ensemble_selective}, checkpoint_path)

    train_rows = [dict(row, zero_shot_margin=float(margins[index]))
                  for index, row in enumerate(pairs) if row["partition"] == "train"]
    dev_baseline_rows = [dict(row, zero_shot_margin=float(margins[index]))
                         for index, row in enumerate(pairs)
                         if row["partition"] == "development"]
    baselines = {
        "zero_shot_margin": baseline(
            train_rows, dev_baseline_rows, "zero_shot_margin"),
        "character_trigram_jaccard": baseline(
            train_rows, dev_baseline_rows, "character_trigram_jaccard"),
    }
    full_gate = {
        "ensemble_constraints_feasible": ensemble_full["constraints_feasible"],
        "all_seed_constraints_feasible": all(
            value["full_binary"]["constraints_feasible"]
            for value in seed_summaries.values()),
        "ensemble_auroc_at_least_point_85":
            ensemble_full["metrics"]["auroc"] >= .85,
        "ensemble_balanced_accuracy_at_least_point_80":
            ensemble_full["metrics"]["balanced_accuracy"] >= .80,
        "all_seed_auroc_at_least_point_85": all(
            value["full_binary"]["metrics"]["auroc"] >= .85
            for value in seed_summaries.values()),
        "all_seed_balanced_accuracy_at_least_point_80": all(
            value["full_binary"]["metrics"]["balanced_accuracy"] >= .80
            for value in seed_summaries.values()),
    }
    selective_gate = {
        "ensemble_selective_gate": ensemble_selective["gate"]["passed"],
        "all_seed_selective_gates": all(
            value["selective"]["gate"]["passed"]
            for value in seed_summaries.values()),
    }
    development_release = all(full_gate.values()) or all(selective_gate.values())
    result = {
        "protocol": PROTOCOL, "rows": len(pairs),
        "train_rows": len(train_indices), "development_rows": len(dev_indices),
        "feature_spec": list(FEATURES), "feature_dim": features.shape[1],
        "seeds": list(SEEDS), "epochs": list(EPOCHS),
        "baselines": baselines, "per_seed": seed_summaries,
        "ensemble": {"full_binary": ensemble_full,
                     "full_binary_grid": ensemble_full_grid,
                     "selective": ensemble_selective,
                     "selective_grid": ensemble_selective_grid},
        "development_gate": {
            "full_binary": {"checks": full_gate, "passed": all(full_gate.values())},
            "selective": {"checks": selective_gate,
                          "passed": all(selective_gate.values())},
            "passed_for_internal_test": development_release,
        },
        "dependencies": dependencies,
        "files_sha256": {predictions_path.name: sha(predictions_path),
                         checkpoint_path.name: sha(checkpoint_path)},
        "resources": {"seconds": time.perf_counter() - started,
                      "gpu": torch.cuda.get_device_name(),
                      "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                      "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30},
        "relation_ids_used_for_supervision_calibration_and_audit_only": True,
        "internal_test_features_loaded": False, "internal_test_scored": False,
        "external_test_features_loaded": False, "external_test_scored": False,
        "sealed_benchmarks_loaded": False,
        "next_stage": ("extract_internal_test_features_once"
                       if development_release else "stop_fewrel_controller_branch"),
        "terminal_status": "completed_fewrel_selective_controller_development",
    }
    write_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-root", required=True)
    parser.add_argument("--embedding-root", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(train(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
