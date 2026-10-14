"""Train the preregistered DocRED semantic relation-space projection."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import time

from tasks.build_fewrel_selective_forgetting import load_jsonl
from tasks.evaluate_natural_semantic_embedding_pairs import (
    binary_metrics, rank_metrics, select_threshold,
)
from tasks.extract_docred_relation_space_embeddings import (
    DATA_AUDIT_SHA256, DATA_MANIFEST_SHA256, load_embeddings,
)
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_fewrel_selective_controller import (
    cluster_false_deletion_upper, relation_bootstrap_draws, score_boundaries,
)


PROTOCOL = "docred_relation_space_projection_development_v1"
EMBEDDING_AUDIT_SHA256 = (
    "8170b5908d6659565bb8a99e71e4f004dff7675f02af0771c8d499980d44a5bc"
)
SEEDS = (23531, 23537, 23543)
EPOCHS = (25, 50, 100)
RANK = 64
TEMPERATURE = .07
MARGIN = .2
STATEMENTS_PER_RELATION = 8
STEPS_PER_EPOCH = 64
PAIR_BATCH_SIZE = 256
LEARNING_RATE = .002
WEIGHT_DECAY = .0001
LOSS_WEIGHTS = {
    "statement_to_description": 1.0,
    "description_to_statement": .5,
    "within_relation": .2,
    "hard_negative_margin": .5,
}


def verify_release(data_root, embedding_root):
    data_root, embedding_root = Path(data_root), Path(embedding_root)
    data_audit = json.loads((data_root / "audit.json").read_text())
    embedding_audit = json.loads((embedding_root / "audit.json").read_text())
    if any((
        sha(data_root / "data/manifest.json") != DATA_MANIFEST_SHA256,
        sha(data_root / "audit.json") != DATA_AUDIT_SHA256,
        sha(embedding_root / "audit.json") != EMBEDDING_AUDIT_SHA256,
        data_audit.get("audit_passed") is not True,
        embedding_audit.get("audit_passed") is not True,
        embedding_audit.get("relation_source_test_feature_extracted") is not False,
        embedding_audit.get("fewrel_internal_test_loaded") is not False,
        embedding_audit.get("fewrel_external_test_loaded") is not False,
        (embedding_root / "runner_exit_code.txt").read_text().strip() != "0",
        (embedding_root / "terminal_status.txt").read_text().strip()
            != "completed_docred_relation_space_embeddings",
    )):
        raise ValueError("projection training requires frozen audited embeddings")
    return {
        "data_manifest_sha256": DATA_MANIFEST_SHA256,
        "data_audit_sha256": DATA_AUDIT_SHA256,
        "embedding_audit_sha256": EMBEDDING_AUDIT_SHA256,
    }


def project(vectors, weight):
    import torch.nn.functional as F
    return F.normalize(F.linear(vectors, weight), p=2, dim=-1)


def supervised_contrastive_loss(vectors, targets, temperature=TEMPERATURE):
    import torch
    similarity = vectors @ vectors.T / temperature
    diagonal = torch.eye(len(vectors), dtype=torch.bool, device=vectors.device)
    similarity = similarity.masked_fill(diagonal, -torch.inf)
    log_probability = similarity - torch.logsumexp(similarity, dim=1, keepdim=True)
    positives = targets[:, None].eq(targets[None, :]) & ~diagonal
    if not bool(positives.any(dim=1).all()):
        raise ValueError("relation-balanced batch lacks a positive statement peer")
    return -(log_probability.masked_fill(~positives, 0).sum(dim=1)
             / positives.sum(dim=1)).mean()


def description_to_statement_loss(logits, targets, relation_count):
    import torch
    log_probability = torch.log_softmax(logits.T, dim=1)
    losses = []
    for relation in range(relation_count):
        mask = targets == relation
        losses.append(-torch.logsumexp(log_probability[relation, mask], dim=0))
    return torch.stack(losses).mean()


def initialized_projection(dimension, seed, device):
    import torch
    generator_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    weight = torch.empty(RANK, dimension, device=device)
    torch.nn.init.orthogonal_(weight)
    torch.random.set_rng_state(generator_state)
    return torch.nn.Parameter(weight)


def train_seed(statement_vectors, statement_relations, description_vectors,
               relation_order, positive_pairs, hard_pairs, seed, device):
    import torch
    import torch.nn.functional as F

    relation_index = {relation: index for index, relation in enumerate(relation_order)}
    by_relation = defaultdict(list)
    for index, relation in enumerate(statement_relations):
        by_relation[relation].append(index)
    if set(by_relation) != set(relation_order):
        raise ValueError("training statement relation coverage changed")
    statements = statement_vectors.to(device)
    descriptions = description_vectors.to(device)
    statement_weight = initialized_projection(statements.shape[1], seed, device)
    description_weight = initialized_projection(descriptions.shape[1], seed + 1, device)
    optimizer = torch.optim.AdamW(
        (statement_weight, description_weight), lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    relation_targets = torch.arange(len(relation_order), device=device).repeat_interleave(
        STATEMENTS_PER_RELATION)
    snapshots, curve = {}, []
    for epoch in range(1, max(EPOCHS) + 1):
        aggregate = defaultdict(float)
        for _ in range(STEPS_PER_EPOCH):
            selected = []
            for relation in relation_order:
                pool = by_relation[relation]
                draw = torch.randint(len(pool), (STATEMENTS_PER_RELATION,),
                                     generator=generator).tolist()
                selected.extend(pool[index] for index in draw)
            selected = torch.tensor(selected, dtype=torch.long, device=device)
            statement_projection = project(statements[selected], statement_weight)
            description_projection = project(descriptions, description_weight)
            logits = statement_projection @ description_projection.T / TEMPERATURE
            statement_loss = F.cross_entropy(logits, relation_targets)
            description_loss = description_to_statement_loss(
                logits, relation_targets, len(relation_order))
            within_loss = supervised_contrastive_loss(
                statement_projection, relation_targets)

            positive_draw = torch.randint(
                len(positive_pairs), (PAIR_BATCH_SIZE,), generator=generator)
            hard_draw = torch.randint(
                len(hard_pairs), (PAIR_BATCH_SIZE,), generator=generator)
            positive = positive_pairs[positive_draw].to(device)
            hard = hard_pairs[hard_draw].to(device)
            positive_score = (project(statements[positive[:, 0]], statement_weight)
                              * project(statements[positive[:, 1]], statement_weight)).sum(1)
            hard_score = (project(statements[hard[:, 0]], statement_weight)
                          * project(statements[hard[:, 1]], statement_weight)).sum(1)
            margin_loss = F.relu(MARGIN + hard_score - positive_score).mean()
            losses = {
                "statement_to_description": statement_loss,
                "description_to_statement": description_loss,
                "within_relation": within_loss,
                "hard_negative_margin": margin_loss,
            }
            loss = sum(LOSS_WEIGHTS[name] * value for name, value in losses.items())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            aggregate["total"] += float(loss.detach())
            for name, value in losses.items():
                aggregate[name] += float(value.detach())
        curve.append({"epoch": epoch, **{
            name: value / STEPS_PER_EPOCH for name, value in aggregate.items()}})
        if epoch in EPOCHS:
            snapshots[epoch] = {
                "statement_weight": statement_weight.detach().cpu().clone(),
                "description_weight": description_weight.detach().cpu().clone(),
            }
    return snapshots, curve


def scores_for_pairs(vectors, pairs, index, weight=None):
    import torch
    left = torch.tensor([index[row["left_instance_id"]] for row in pairs])
    right = torch.tensor([index[row["right_instance_id"]] for row in pairs])
    values = vectors if weight is None else project(vectors, weight)
    return (values[left] * values[right]).sum(dim=1).tolist()


def metric_set(labels, scores):
    threshold, values, safety_feasible = select_threshold(labels, scores)
    values.update(rank_metrics(labels, scores))
    values["false_deletion_rate"] = 1 - values["negative_specificity"]
    return {"threshold": threshold, "full_safety_feasible": safety_feasible,
            "metrics": values}


def selective_operating_point(labels, scores, relations):
    draw_spec = relation_bootstrap_draws(relations)
    candidates = []
    for value in score_boundaries(labels, scores):
        predictions = [int(score >= value["threshold"]) for score in scores]
        upper = cluster_false_deletion_upper(labels, predictions, relations, draw_spec)
        candidates.append({**value,
                           "false_deletion_relation_bootstrap_upper_95": upper})
    feasible = [value for value in candidates if all((
        value["deletion_precision"] >= .90,
        value["positive_recall"] >= .25,
        value["false_deletion_relation_bootstrap_upper_95"] <= .05,
    ))]
    pool = feasible or candidates
    selected = max(pool, key=lambda value: (
        value in feasible, value["positive_recall"], value["deletion_precision"],
        -value["false_deletion_relation_bootstrap_upper_95"]))
    return {"feasible": bool(feasible), "selected": selected}


def relation_metrics(pairs, labels, scores):
    output = {}
    for relation in sorted({row["left_relation_id"] for row in pairs}):
        indices = [index for index, row in enumerate(pairs)
                   if row["left_relation_id"] == relation]
        local_labels = [labels[index] for index in indices]
        local_scores = [scores[index] for index in indices]
        output[relation] = rank_metrics(local_labels, local_scores)
    return output


def train(args):
    import torch

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    dependencies = verify_release(args.data_root, args.embedding_root)
    data_root = Path(args.data_root)
    manifest = json.loads((data_root / "data/manifest.json").read_text())
    all_pairs = load_jsonl(data_root / "data/pairs.jsonl")
    train_pairs = [row for row in all_pairs if row["partition"] == "representation_train"]
    dev_pairs = [row for row in all_pairs if row["partition"] == "relation_development"]
    records, vectors = load_embeddings(args.embedding_root)
    statement_rows = [(row, vectors[index]) for index, row in enumerate(records)
                      if row["item_type"] == "statement"]
    description_rows = [(row, vectors[index]) for index, row in enumerate(records)
                        if row["item_type"] == "relation_description"]
    train_rows = [(row, vector) for row, vector in statement_rows
                  if row["partition"] == "representation_train"]
    dev_rows = [(row, vector) for row, vector in statement_rows
                if row["partition"] == "relation_development"]
    relation_order = manifest["partitions"]["representation_train"]
    descriptions = {row["relation_id"]: vector for row, vector in description_rows}
    if set(descriptions) != set(relation_order):
        raise ValueError("description embedding relation coverage mismatch")
    train_vectors = torch.stack([value for _, value in train_rows]).float()
    train_relations = [row["relation_id"] for row, _ in train_rows]
    description_vectors = torch.stack([descriptions[value] for value in relation_order]).float()
    train_index = {row["item_id"]: index for index, (row, _) in enumerate(train_rows)}
    dev_vectors = torch.stack([value for _, value in dev_rows]).float()
    dev_index = {row["item_id"]: index for index, (row, _) in enumerate(dev_rows)}
    positive_pairs = torch.tensor([
        [train_index[row["left_instance_id"]], train_index[row["right_instance_id"]]]
        for row in train_pairs if row["kind"] == "same_relation_positive"],
        dtype=torch.long)
    hard_pairs = torch.tensor([
        [train_index[row["left_instance_id"]], train_index[row["right_instance_id"]]]
        for row in train_pairs if row["kind"] == "lexical_hard_negative"],
        dtype=torch.long)
    labels = [int(row["same_relation"]) for row in dev_pairs]
    relations = [row["left_relation_id"] for row in dev_pairs]
    raw_scores = scores_for_pairs(dev_vectors, dev_pairs, dev_index)
    raw = metric_set(labels, raw_scores)
    trigram_scores = [float(row["character_trigram_jaccard"]) for row in dev_pairs]
    trigram = metric_set(labels, trigram_scores)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device.index or 0)
        torch.cuda.reset_peak_memory_stats()
        torch.backends.cuda.matmul.allow_tf32 = False
    started = time.perf_counter()
    seed_summaries, seed_checkpoints, selected_scores = {}, {}, []
    for seed in SEEDS:
        snapshots, curve = train_seed(
            train_vectors, train_relations, description_vectors, relation_order,
            positive_pairs, hard_pairs, seed, device)
        epoch_results = []
        for epoch in EPOCHS:
            scores = scores_for_pairs(
                dev_vectors, dev_pairs, dev_index,
                snapshots[epoch]["statement_weight"])
            epoch_results.append({"epoch": epoch, "scores": scores,
                                  "rank_metrics": rank_metrics(labels, scores)})
        selected = max(epoch_results, key=lambda value: (
            value["rank_metrics"]["auroc"],
            value["rank_metrics"]["average_precision"], -value["epoch"]))
        metrics = metric_set(labels, selected["scores"])
        selected_scores.append(torch.tensor(selected["scores"]))
        seed_summaries[str(seed)] = {
            "selected_epoch": selected["epoch"],
            "epoch_rank_metrics": [{"epoch": row["epoch"],
                                    "rank_metrics": row["rank_metrics"]}
                                   for row in epoch_results],
            "selected_metrics": metrics, "training_curve": curve,
        }
        seed_checkpoints[str(seed)] = {
            "selected_epoch": selected["epoch"], **snapshots[selected["epoch"]]}

    ensemble_scores = torch.stack(selected_scores).mean(dim=0).tolist()
    ensemble = metric_set(labels, ensemble_scores)
    selective = selective_operating_point(labels, ensemble_scores, relations)
    per_relation = relation_metrics(dev_pairs, labels, ensemble_scores)
    relation_macro = {
        name: sum(value[name] for value in per_relation.values()) / len(per_relation)
        for name in ("auroc", "average_precision")}
    hard_indices = [index for index, row in enumerate(dev_pairs)
                    if row["kind"] == "lexical_hard_negative"]
    hard_specificity = sum(ensemble_scores[index] < ensemble["threshold"]
                           for index in hard_indices) / len(hard_indices)
    gate_checks = {
        "ensemble_auroc_at_least_point_85": ensemble["metrics"]["auroc"] >= .85,
        "ensemble_balanced_accuracy_at_least_point_80": (
            ensemble["metrics"]["balanced_accuracy"] >= .80),
        "every_seed_auroc_at_least_point_82": all(
            value["selected_metrics"]["metrics"]["auroc"] >= .82
            for value in seed_summaries.values()),
        "full_or_selective_safety": (
            ensemble["full_safety_feasible"] or selective["feasible"]),
        "balanced_accuracy_gain_over_raw_at_least_point_05": (
            ensemble["metrics"]["balanced_accuracy"]
            - raw["metrics"]["balanced_accuracy"] >= .05),
        "every_relation_auroc_at_least_point_60": min(
            value["auroc"] for value in per_relation.values()) >= .60,
    }
    performance_gate = {"checks": gate_checks, "passed": all(gate_checks.values())}

    predictions_path = output / "development_predictions.jsonl"
    with predictions_path.open("x") as sink:
        for index, pair in enumerate(dev_pairs):
            sink.write(json.dumps({
                "pair_id": pair["pair_id"], "kind": pair["kind"],
                "left_relation_id": pair["left_relation_id"],
                "right_relation_id": pair["right_relation_id"],
                "same_relation": labels[index], "raw_cosine": raw_scores[index],
                "character_trigram_jaccard": trigram_scores[index],
                "ensemble_projected_cosine": ensemble_scores[index],
            }, sort_keys=True) + "\n")
    checkpoint_path = output / "projection_checkpoints.pt"
    torch.save({
        "protocol": PROTOCOL, "rank": RANK, "relation_order": relation_order,
        "seeds": list(SEEDS), "controllers": seed_checkpoints,
        "ensemble_threshold": ensemble["threshold"],
        "selective_operating_point": selective,
    }, checkpoint_path)
    summary = {
        "protocol": PROTOCOL, "dependencies": dependencies,
        "hyperparameters": {
            "rank": RANK, "temperature": TEMPERATURE, "margin": MARGIN,
            "statements_per_relation": STATEMENTS_PER_RELATION,
            "steps_per_epoch": STEPS_PER_EPOCH, "pair_batch_size": PAIR_BATCH_SIZE,
            "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
            "loss_weights": LOSS_WEIGHTS, "seeds": list(SEEDS),
            "epochs": list(EPOCHS),
        },
        "counts": {"train_statements": len(train_rows),
                   "training_relations": len(relation_order),
                   "train_pairs": len(train_pairs), "development_statements": len(dev_rows),
                   "development_relations": len(set(relations)),
                   "development_pairs": len(dev_pairs)},
        "baselines": {"raw_statement_cosine": raw,
                      "character_trigram_jaccard": trigram},
        "seeds": seed_summaries,
        "ensemble": ensemble, "selective": selective,
        "relation_macro": relation_macro, "per_relation": per_relation,
        "lexical_hard_negative_specificity": hard_specificity,
        "development_performance_gate": performance_gate,
        "independent_audit_pending": True,
        "next_stage": ("independent_audit_then_relation_source_test"
                       if performance_gate["passed"] else "stop_representation_branch"),
        "files_sha256": {"development_predictions.jsonl": sha(predictions_path),
                         "projection_checkpoints.pt": sha(checkpoint_path)},
        "seconds": time.perf_counter() - started,
        "peak_allocated_gib": (torch.cuda.max_memory_allocated() / 2**30
                               if device.type == "cuda" else 0.0),
        "relation_source_test_feature_extracted": False,
        "fewrel_internal_test_loaded": False, "fewrel_external_test_loaded": False,
        "sealed_benchmarks_loaded": False,
        "terminal_status": "completed_docred_relation_space_projection_development",
    }
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--embedding-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    result = train(parser.parse_args())
    print(json.dumps({"status": result["terminal_status"],
                      "gate": result["development_performance_gate"],
                      "ensemble": result["ensemble"]["metrics"],
                      "next_stage": result["next_stage"]}, indent=2))


if __name__ == "__main__":
    main()
