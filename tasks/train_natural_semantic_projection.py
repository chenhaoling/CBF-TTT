"""Train nested-OOF rank-16 semantic projections on frozen Qwen surfaces."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import statistics
import time

from tasks.build_natural_semantic_relation_pairs import load_jsonl
from tasks.evaluate_natural_semantic_embedding_pairs import (
    FOLDS,
    PAIR_MANIFEST_SHA256,
    PAIR_ROWS_SHA256,
    binary_metrics,
    load_embeddings,
    rank_metrics,
    select_threshold,
)
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "natural_semantic_rank16_projection_v1"
EMBEDDING_AUDIT_SHA256 = (
    "4ff61768ac749b6479f9209acb39b04463fb25b6e48a7d3aa811ca223e981649"
)
RAW_BASELINE_SUMMARY_SHA256 = (
    "3d51630c668a692aa8ac1d822cfe001aa6e9d96c2261d3c42e860393d9344d37"
)
MULTIVIEW_AUDIT_SHA256 = (
    "ea817bab098ce41446309fa79a76f5eb4cc6afa6037a78c481f775e8057f28fb"
)
MULTIVIEW_PAIR_SUMMARY_SHA256 = (
    "f8e2ba5a21dd8343278f54627444c02be0d6e59a72a777451feaaf6aa9f4b999"
)
SEEDS = (22031, 22037, 22043)
EPOCHS = (25, 50, 100)
RANK = 16
TEMPERATURE = .07
MARGIN = .2
HARD_NEGATIVE_WEIGHT = .5
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0
RELATION_BALANCED_STEPS = 128


def make_projection(dim, seed):
    import torch
    torch.manual_seed(seed)
    layer = torch.nn.Linear(dim, RANK, bias=False)
    torch.nn.init.orthogonal_(layer.weight)
    return layer


def project(layer, values):
    import torch.nn.functional as F
    return F.normalize(layer(values), p=2, dim=-1)


def build_relation_triples(pairs, surface_index, folds):
    hard = {(int(row["outer_fold"]), row["anchor_source_row_sha256"]): row
            for row in pairs if row["kind"] == "lexical_hard_negative"
            and int(row["outer_fold"]) in folds}
    grouped = {}
    for row in pairs:
        fold = int(row["outer_fold"])
        if fold not in folds or not row["same_relation"]:
            continue
        negative = hard.get((fold, row["anchor_source_row_sha256"]))
        if negative is None:
            raise ValueError("positive pair lacks registered lexical hard negative")
        relation = row["anchor_relation_id"]
        grouped.setdefault(relation, []).append((
            surface_index[row["left_surface_id"]],
            surface_index[row["right_surface_id"]],
            surface_index[negative["right_surface_id"]],
        ))
    grouped = {key: sorted(set(values)) for key, values in sorted(grouped.items())}
    if len(grouped) < 2 or any(not values for values in grouped.values()):
        raise ValueError("projection training needs at least two populated relations")
    return grouped


def epoch_indices(grouped, seed, epoch, steps, device):
    import torch
    columns = []
    for relation_index, values in enumerate(grouped.values()):
        generator = torch.Generator().manual_seed(
            seed * 1_000_003 + epoch * 10_007 + relation_index * 101)
        selected = torch.randint(len(values), (steps,), generator=generator).tolist()
        columns.append([values[index] for index in selected])
    left = torch.tensor([[column[step][0] for column in columns]
                         for step in range(steps)], device=device)
    positive = torch.tensor([[column[step][1] for column in columns]
                             for step in range(steps)], device=device)
    negative = torch.tensor([[column[step][2] for column in columns]
                             for step in range(steps)], device=device)
    return left, positive, negative


def train_projection(embeddings, grouped, seed):
    import torch
    import torch.nn.functional as F
    layer = make_projection(embeddings.shape[1], seed).to(embeddings.device)
    optimizer = torch.optim.AdamW(layer.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    checkpoints, curve = {}, []
    relation_count = len(grouped)
    targets = torch.arange(relation_count, device=embeddings.device) \
        .repeat(RELATION_BALANCED_STEPS)
    for epoch in range(1, max(EPOCHS) + 1):
        left, positive, negative = epoch_indices(
            grouped, seed, epoch, RELATION_BALANCED_STEPS, embeddings.device)
        optimizer.zero_grad(set_to_none=True)
        left_value = project(layer, embeddings[left])
        positive_value = project(layer, embeddings[positive])
        negative_value = project(layer, embeddings[negative])
        logits = torch.einsum("brd,bsd->brs", left_value, positive_value) \
            / TEMPERATURE
        info_nce = (F.cross_entropy(logits.reshape(-1, relation_count), targets)
                    + F.cross_entropy(
                        logits.transpose(1, 2).reshape(-1, relation_count), targets)) / 2
        positive_cosine = (left_value * positive_value).sum(-1)
        negative_cosine = (left_value * negative_value).sum(-1)
        hard_loss = F.relu(negative_cosine - positive_cosine + MARGIN).mean()
        loss = info_nce + HARD_NEGATIVE_WEIGHT * hard_loss
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(
            layer.parameters(), GRADIENT_CLIP))
        optimizer.step()
        if epoch in EPOCHS:
            checkpoints[epoch] = {
                key: value.detach().cpu().clone()
                for key, value in layer.state_dict().items()}
            curve.append({
                "epoch": epoch, "loss": float(loss),
                "info_nce": float(info_nce), "hard_negative_loss": float(hard_loss),
                "gradient_norm_before_clip": gradient_norm,
                "mean_positive_cosine": float(positive_cosine.mean()),
                "mean_hard_negative_cosine": float(negative_cosine.mean()),
            })
    return checkpoints, curve


def projection_scores(embeddings, pairs, surface_index, state):
    import torch
    layer = make_projection(embeddings.shape[1], 0).to(embeddings.device)
    layer.load_state_dict(state)
    with torch.inference_mode():
        values = project(layer, embeddings)
        left = torch.tensor([surface_index[row["left_surface_id"]] for row in pairs],
                            device=embeddings.device)
        right = torch.tensor([surface_index[row["right_surface_id"]] for row in pairs],
                             device=embeddings.device)
        scores = (values[left] * values[right]).sum(-1).cpu().tolist()
    return scores


def metric_with_ranking(labels, scores, threshold):
    metric = binary_metrics(labels, [int(score >= threshold) for score in scores])
    metric.update(rank_metrics(labels, scores))
    return metric


def select_epoch(candidates):
    return max(candidates, key=lambda value: (
        value["inner_constraints_feasible"],
        value["inner_metrics"]["balanced_accuracy"],
        value["inner_metrics"]["f1"],
        value["inner_metrics"]["negative_specificity"],
        -value["epoch"],
    ))


def verify_dependencies(pair_root, embedding_root, raw_root):
    pair_root, embedding_root, raw_root = map(Path, (
        pair_root, embedding_root, raw_root))
    embedding_audit = json.loads((embedding_root / "audit.json").read_text())
    raw_summary = json.loads((raw_root / "results/summary.json").read_text())
    checks = (
        sha(pair_root / "manifest.json") == PAIR_MANIFEST_SHA256,
        sha(pair_root / "pairs.jsonl") == PAIR_ROWS_SHA256,
        sha(embedding_root / "audit.json") == EMBEDDING_AUDIT_SHA256,
        embedding_audit.get("audit_passed") is True,
        (embedding_root / "runner_exit_code.txt").read_text().strip() == "0",
        sha(raw_root / "results/summary.json") == RAW_BASELINE_SUMMARY_SHA256,
        raw_summary.get("terminal_status")
            == "completed_natural_semantic_raw_embedding_pair_baseline",
        (raw_root / "runner_exit_code.txt").read_text().strip() == "0",
    )
    if not all(checks):
        raise ValueError("projection training requires frozen audited dependencies")
    return {
        "pair_manifest_sha256": sha(pair_root / "manifest.json"),
        "pairs_sha256": sha(pair_root / "pairs.jsonl"),
        "embedding_audit_sha256": sha(embedding_root / "audit.json"),
        "raw_baseline_summary_sha256": sha(raw_root / "results/summary.json"),
    }, raw_summary


def verify_multiview_dependencies(pair_root, embedding_root, raw_root):
    from tasks.extract_natural_semantic_multiview_embeddings import VIEWS
    pair_root, embedding_root, raw_root = map(Path, (
        pair_root, embedding_root, raw_root))
    audit = json.loads((embedding_root / "audit.json").read_text())
    raw_summary = json.loads((raw_root / "results/summary.json").read_text())
    checks = (
        sha(pair_root / "manifest.json") == PAIR_MANIFEST_SHA256,
        sha(pair_root / "pairs.jsonl") == PAIR_ROWS_SHA256,
        sha(embedding_root / "audit.json") == MULTIVIEW_AUDIT_SHA256,
        audit.get("audit_passed") is True,
        audit.get("views") == list(VIEWS),
        (embedding_root / "runner_exit_code.txt").read_text().strip() == "0",
        sha(raw_root / "results/summary.json") == MULTIVIEW_PAIR_SUMMARY_SHA256,
        raw_summary.get("terminal_status")
            == "completed_natural_semantic_multiview_pair_diagnostic",
        (raw_root / "runner_exit_code.txt").read_text().strip() == "0",
    )
    if not all(checks):
        raise ValueError("multiview projection requires frozen audited dependencies")
    return {
        "pair_manifest_sha256": sha(pair_root / "manifest.json"),
        "pairs_sha256": sha(pair_root / "pairs.jsonl"),
        "multiview_audit_sha256": sha(embedding_root / "audit.json"),
        "multiview_pair_summary_sha256": sha(raw_root / "results/summary.json"),
    }, raw_summary


def train(args):
    import torch
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    if args.embedding_view == "mean":
        dependencies, raw_summary = verify_dependencies(
            args.pair_root, args.embedding_root, args.raw_root)
        surface_ids, embedding_cpu = load_embeddings(args.embedding_root)
        raw_balanced = raw_summary["aggregate"]["balanced_accuracy"]
        protocol = PROTOCOL
    else:
        from tasks.extract_natural_semantic_multiview_embeddings import (
            load_multiview_embeddings,
        )
        dependencies, raw_summary = verify_multiview_dependencies(
            args.pair_root, args.embedding_root, args.raw_root)
        surface_ids, views = load_multiview_embeddings(args.embedding_root)
        embedding_cpu = views[args.embedding_view]
        raw_balanced = raw_summary["nested_selected_oof"]["balanced_accuracy"]
        protocol = f"natural_semantic_rank16_{args.embedding_view}_projection_v2"
    pairs = load_jsonl(Path(args.pair_root) / "pairs.jsonl")
    surface_index = {value: index for index, value in enumerate(surface_ids)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(2201)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    embeddings = embedding_cpu.to(device)
    labels = [int(row["same_relation"]) for row in pairs]
    started = time.perf_counter()
    selections, checkpoint_folds, output_rows = {}, {}, []
    per_seed_records = {str(seed): [] for seed in SEEDS}
    fold_ensemble_metrics = {}

    for outer in range(FOLDS):
        inner = (outer + 1) % FOLDS
        train_folds = sorted(set(range(FOLDS)) - {outer, inner})
        triples = build_relation_triples(pairs, surface_index, set(train_folds))
        inner_pairs = [row for row in pairs if int(row["outer_fold"]) == inner]
        outer_pairs = [row for row in pairs if int(row["outer_fold"]) == outer]
        inner_labels = [int(row["same_relation"]) for row in inner_pairs]
        outer_labels = [int(row["same_relation"]) for row in outer_pairs]
        seed_artifacts, inner_selected_scores, outer_selected_scores = {}, [], []

        for seed in SEEDS:
            checkpoints, curve = train_projection(embeddings, triples, seed + outer * 1000)
            candidates = []
            for epoch in EPOCHS:
                scores = projection_scores(
                    embeddings, inner_pairs, surface_index, checkpoints[epoch])
                threshold, metric, feasible = select_threshold(inner_labels, scores)
                metric.update(rank_metrics(inner_labels, scores))
                candidates.append({
                    "epoch": epoch, "threshold": threshold,
                    "inner_constraints_feasible": feasible,
                    "inner_metrics": metric, "scores": scores,
                })
            selected = select_epoch(candidates)
            state = checkpoints[selected["epoch"]]
            outer_scores = projection_scores(
                embeddings, outer_pairs, surface_index, state)
            outer_metric = metric_with_ranking(
                outer_labels, outer_scores, selected["threshold"])
            inner_selected_scores.append(selected["scores"])
            outer_selected_scores.append(outer_scores)
            per_seed_records[str(seed)].extend(zip(
                outer_labels, outer_scores,
                [selected["threshold"]] * len(outer_labels)))
            seed_artifacts[str(seed)] = {
                "seed": seed, "effective_seed": seed + outer * 1000,
                "selected_epoch": selected["epoch"],
                "selected_threshold": selected["threshold"],
                "inner_constraints_feasible": selected["inner_constraints_feasible"],
                "inner_metrics": selected["inner_metrics"],
                "outer_metrics": outer_metric,
                "candidate_metrics": [{key: value for key, value in row.items()
                                       if key != "scores"} for row in candidates],
                "training_curve": curve,
                "state": state,
            }

        inner_ensemble = [statistics.fmean(values) for values in
                          zip(*inner_selected_scores)]
        outer_ensemble = [statistics.fmean(values) for values in
                          zip(*outer_selected_scores)]
        ensemble_threshold, ensemble_inner_metric, ensemble_feasible = \
            select_threshold(inner_labels, inner_ensemble)
        ensemble_inner_metric.update(rank_metrics(inner_labels, inner_ensemble))
        ensemble_outer_metric = metric_with_ranking(
            outer_labels, outer_ensemble, ensemble_threshold)
        fold_ensemble_metrics[str(outer)] = ensemble_outer_metric
        selections[str(outer)] = {
            "outer_fold": outer, "inner_fold": inner, "train_folds": train_folds,
            "training_relations": len(triples),
            "training_triples": sum(map(len, triples.values())),
            "ensemble_threshold": ensemble_threshold,
            "ensemble_inner_constraints_feasible": ensemble_feasible,
            "ensemble_inner_metrics": ensemble_inner_metric,
            "ensemble_outer_metrics": ensemble_outer_metric,
            "seeds": {seed: {key: value for key, value in artifact.items()
                             if key != "state"}
                      for seed, artifact in seed_artifacts.items()},
        }
        checkpoint_folds[str(outer)] = {
            "inner_fold": inner, "train_folds": train_folds,
            "ensemble_threshold": ensemble_threshold,
            "seeds": seed_artifacts,
        }
        for position, (row, score) in enumerate(zip(outer_pairs, outer_ensemble)):
            seed_scores = {str(seed): outer_selected_scores[index][position]
                           for index, seed in enumerate(SEEDS)}
            output_rows.append({
                "pair_id": row["pair_id"], "outer_fold": outer,
                "kind": row["kind"], "same_relation": int(row["same_relation"]),
                "projected_cosine_by_seed": seed_scores,
                "ensemble_projected_cosine": score,
                "ensemble_threshold": ensemble_threshold,
                "ensemble_prediction": int(score >= ensemble_threshold),
            })

    output_rows.sort(key=lambda row: row["pair_id"])
    ensemble = binary_metrics(
        [row["same_relation"] for row in output_rows],
        [row["ensemble_prediction"] for row in output_rows])
    ensemble.update(rank_metrics(
        [row["same_relation"] for row in output_rows],
        [row["ensemble_projected_cosine"] for row in output_rows]))
    per_seed = {}
    for seed, records in per_seed_records.items():
        seed_labels = [value[0] for value in records]
        seed_scores = [value[1] for value in records]
        seed_predictions = [int(value[1] >= value[2]) for value in records]
        per_seed[seed] = binary_metrics(seed_labels, seed_predictions)
        per_seed[seed].update(rank_metrics(seed_labels, seed_scores))
    by_kind = {}
    for kind in sorted({row["kind"] for row in output_rows}):
        selected = [row for row in output_rows if row["kind"] == kind]
        by_kind[kind] = binary_metrics(
            [row["same_relation"] for row in selected],
            [row["ensemble_prediction"] for row in selected])
    checks = {
        "complete_nested_oof_coverage": (
            len(output_rows) == len(pairs)
            and len({row["pair_id"] for row in output_rows}) == len(pairs)),
        "all_inner_seed_and_ensemble_constraints_feasible": all(
            value["ensemble_inner_constraints_feasible"]
            and all(seed["inner_constraints_feasible"]
                    for seed in value["seeds"].values())
            for value in selections.values()),
        "ensemble_balanced_accuracy_gain_over_raw_at_least_10pp":
            ensemble["balanced_accuracy"] - raw_balanced >= .10,
        "all_outer_fold_auroc_at_least_point_75": all(
            value["auroc"] >= .75 for value in fold_ensemble_metrics.values()),
        "all_outer_fold_balanced_accuracy_at_least_point_65": all(
            value["balanced_accuracy"] >= .65
            for value in fold_ensemble_metrics.values()),
        "all_seed_auroc_at_least_point_70": all(
            value["auroc"] >= .70 for value in per_seed.values()),
    }
    rows_path = output / "oof_projected_cosine.jsonl"
    with rows_path.open("x") as sink:
        for row in output_rows:
            sink.write(json.dumps(row) + "\n")
    checkpoint_path = output / "projection_checkpoints.pt"
    torch.save({
        "protocol": protocol, "embedding_view": args.embedding_view,
        "rank": RANK, "seeds": SEEDS,
        "embedding_dim": embeddings.shape[1], "folds": checkpoint_folds,
    }, checkpoint_path)
    result = {
        "protocol": protocol,
        "embedding_view": args.embedding_view,
        "rows": len(output_rows),
        "ensemble_oof": ensemble,
        "per_seed_oof": per_seed,
        "per_fold_ensemble": fold_ensemble_metrics,
        "by_kind": by_kind,
        "selection": selections,
        "raw_baseline_balanced_accuracy": raw_balanced,
        "ensemble_balanced_accuracy_gain_over_raw":
            ensemble["balanced_accuracy"] - raw_balanced,
        "projection_gate": {"checks": checks, "passed": all(checks.values())},
        "dependencies": dependencies,
        "hyperparameters": {
            "rank": RANK, "seeds": list(SEEDS), "epochs": list(EPOCHS),
            "temperature": TEMPERATURE, "margin": MARGIN,
            "hard_negative_weight": HARD_NEGATIVE_WEIGHT,
            "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
            "gradient_clip": GRADIENT_CLIP,
            "relation_balanced_steps_per_epoch": RELATION_BALANCED_STEPS,
        },
        "files_sha256": {
            rows_path.name: sha(rows_path),
            checkpoint_path.name: sha(checkpoint_path),
        },
        "resources": {
            "seconds": time.perf_counter() - started,
            "device": str(device),
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "peak_allocated_gib": (torch.cuda.max_memory_allocated() / 2**30
                                   if torch.cuda.is_available() else 0),
            "peak_reserved_gib": (torch.cuda.max_memory_reserved() / 2**30
                                  if torch.cuda.is_available() else 0),
        },
        "relation_ids_used_as_training_labels_only": True,
        "model_outcomes_loaded": False,
        "sealed_benchmarks_loaded": False,
        "next_stage": ("train_nested_semantic_residual_controller"
                       if all(checks.values())
                       else "diagnose_semantic_projection_failure"),
        "terminal_status": "completed_natural_semantic_rank16_projection",
    }
    write_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-root", required=True)
    parser.add_argument("--embedding-root", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--embedding-view", default="mean",
                        choices=("mean", "last_token", "last8_mean",
                                 "subject_context8_mean"))
    result = train(parser.parse_args())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
