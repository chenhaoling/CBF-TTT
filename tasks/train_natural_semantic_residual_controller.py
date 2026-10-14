"""Train nested-OOF utility/risk residual controllers on semantic pairs."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import statistics
import time

from tasks.build_natural_semantic_relation_pairs import load_jsonl
from tasks.evaluate_natural_semantic_embedding_pairs import (
    FOLDS, PAIR_MANIFEST_SHA256, PAIR_ROWS_SHA256,
    binary_metrics, rank_metrics,
)
from tasks.extract_natural_semantic_multiview_embeddings import (
    VIEWS, load_multiview_embeddings,
)
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_natural_semantic_projection import (
    EPOCHS, SEEDS, make_projection, project,
)


PROTOCOL = "natural_semantic_residual_controller_v1"
MULTIVIEW_AUDIT_SHA256 = (
    "ea817bab098ce41446309fa79a76f5eb4cc6afa6037a78c481f775e8057f28fb"
)
PROJECTION_SUMMARY_SHA256 = (
    "308d04379a3ec761efe0b1563b0288774d89345cc197054b4bb18e735e2daaf7"
)
FEATURE_NAMES = (
    "mean_cosine", "last_token_cosine", "last8_cosine",
    "subject_context8_cosine", "projected_last_token_cosine",
    "character_trigram_jaccard",
)
HIDDEN = 64
THRESHOLDS = (.4, .5, .6)
RISK_PENALTIES = (0.0, .5, 1.0, 2.0)
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0


def make_controller(dim, seed):
    import torch
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(dim, HIDDEN), torch.nn.LayerNorm(HIDDEN), torch.nn.GELU(),
        torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.LayerNorm(HIDDEN), torch.nn.GELU(),
        torch.nn.Linear(HIDDEN, 2),
    )


def base_cosines(views, pairs, surface_index):
    import torch
    left = torch.tensor([surface_index[row["left_surface_id"]] for row in pairs])
    right = torch.tensor([surface_index[row["right_surface_id"]] for row in pairs])
    return {view: (matrix[left] * matrix[right]).sum(dim=1)
            for view, matrix in views.items()}


def projected_cosines(matrix, pairs, surface_index, state):
    import torch
    layer = make_projection(matrix.shape[1], 0)
    layer.load_state_dict(state)
    with torch.inference_mode():
        values = project(layer, matrix)
        left = torch.tensor([surface_index[row["left_surface_id"]] for row in pairs])
        right = torch.tensor([surface_index[row["right_surface_id"]] for row in pairs])
        return (values[left] * values[right]).sum(dim=1)


def feature_matrix(raw, projected, pairs):
    import torch
    return torch.stack([
        raw["mean"], raw["last_token"], raw["last8_mean"],
        raw["subject_context8_mean"], projected,
        torch.tensor([float(row["character_trigram_jaccard"]) for row in pairs]),
    ], dim=1).float()


def train_controller(features, labels, train_indices, seed):
    import torch
    x = features[train_indices]
    y = labels[train_indices]
    mean = x.mean(dim=0); std = x.std(dim=0, unbiased=False).clamp_min(1e-4)
    x = ((x - mean) / std).cuda(); y = y.cuda()
    model = make_controller(x.shape[1], seed).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    snapshots, curve = {}, []
    for epoch in range(1, max(EPOCHS) + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        targets = torch.stack((y, 1 - y), dim=1)
        utility_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits[:, 0], targets[:, 0])
        risk_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits[:, 1], targets[:, 1])
        loss = utility_loss + risk_loss
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(
            model.parameters(), GRADIENT_CLIP))
        optimizer.step()
        if epoch in EPOCHS:
            snapshots[epoch] = {key: value.detach().cpu().clone()
                                for key, value in model.state_dict().items()}
            curve.append({"epoch": epoch, "loss": float(loss.detach()),
                          "utility_loss": float(utility_loss.detach()),
                          "risk_loss": float(risk_loss.detach()),
                          "gradient_norm_before_clip": gradient_norm})
    return snapshots, mean, std, curve


def controller_probabilities(features, indices, state, mean, std, seed):
    import torch
    model = make_controller(features.shape[1], seed)
    model.load_state_dict(state); model.eval()
    with torch.inference_mode():
        return torch.sigmoid(model((features[indices] - mean) / std))


def decision_metric(labels, probabilities, threshold, risk_penalty):
    score = probabilities[:, 0] - risk_penalty * probabilities[:, 1]
    predictions = (score >= threshold).long().tolist()
    values = binary_metrics(labels.tolist(), predictions)
    values.update(rank_metrics(labels.tolist(), score.tolist()))
    values["risk_violation"] = 1 - values["negative_specificity"]
    return values, score.tolist(), predictions


def search_decision(labels, probabilities):
    candidates = []
    for risk_penalty in RISK_PENALTIES:
        for threshold in THRESHOLDS:
            metric, _, _ = decision_metric(
                labels, probabilities, threshold, risk_penalty)
            feasible = (metric["positive_recall"] >= .80
                        and metric["negative_specificity"] >= .95
                        and metric["risk_violation"] <= .03)
            candidates.append({"threshold": threshold, "risk_penalty": risk_penalty,
                               "constraints_feasible": feasible, "metrics": metric})
    selected = max(candidates, key=lambda value: (
        value["constraints_feasible"], value["metrics"]["balanced_accuracy"],
        value["metrics"]["positive_recall"],
        value["metrics"]["negative_specificity"],
        -value["risk_penalty"], -value["threshold"]))
    return selected, candidates


def select_snapshot(candidates):
    return max(candidates, key=lambda value: (
        value["selection"]["constraints_feasible"],
        value["selection"]["metrics"]["balanced_accuracy"],
        value["selection"]["metrics"]["positive_recall"],
        value["selection"]["metrics"]["negative_specificity"],
        -value["epoch"]))


def verify_release(pair_root, embedding_root, projection_root):
    pair_root, embedding_root, projection_root = map(Path, (
        pair_root, embedding_root, projection_root))
    audit = json.loads((embedding_root / "audit.json").read_text())
    projection = json.loads((projection_root / "results/summary.json").read_text())
    checks = (
        sha(pair_root / "manifest.json") == PAIR_MANIFEST_SHA256,
        sha(pair_root / "pairs.jsonl") == PAIR_ROWS_SHA256,
        sha(embedding_root / "audit.json") == MULTIVIEW_AUDIT_SHA256,
        audit.get("audit_passed") is True,
        sha(projection_root / "results/summary.json") == PROJECTION_SUMMARY_SHA256,
        projection.get("embedding_view") == "last_token",
        projection.get("terminal_status") == "completed_natural_semantic_rank16_projection",
        projection["files_sha256"]["projection_checkpoints.pt"]
            == sha(projection_root / "results/projection_checkpoints.pt"),
        (projection_root / "runner_exit_code.txt").read_text().strip() == "0",
    )
    if not all(checks):
        raise ValueError("controller requires frozen multiview/projection releases")
    return {"pair_manifest_sha256": sha(pair_root / "manifest.json"),
            "pairs_sha256": sha(pair_root / "pairs.jsonl"),
            "multiview_audit_sha256": sha(embedding_root / "audit.json"),
            "projection_summary_sha256": sha(projection_root / "results/summary.json"),
            "projection_checkpoint_sha256":
                sha(projection_root / "results/projection_checkpoints.pt")}


def train(args):
    import torch
    output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
    dependencies = verify_release(
        args.pair_root, args.embedding_root, args.projection_root)
    pairs = load_jsonl(Path(args.pair_root) / "pairs.jsonl")
    surface_ids, views = load_multiview_embeddings(args.embedding_root)
    surface_index = {value: index for index, value in enumerate(surface_ids)}
    raw = base_cosines(views, pairs, surface_index)
    labels = torch.tensor([float(row["same_relation"]) for row in pairs])
    checkpoint = torch.load(
        Path(args.projection_root) / "results/projection_checkpoints.pt",
        map_location="cpu", weights_only=True)
    if checkpoint.get("embedding_view") != "last_token":
        raise ValueError("wrong projection representation")
    torch.manual_seed(2221)
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
            projection_artifact = checkpoint["folds"][str(outer)]["seeds"][str(seed)]
            projected = projected_cosines(
                views["last_token"], pairs, surface_index,
                projection_artifact["state"])
            features = feature_matrix(raw, projected, pairs)
            effective_seed = seed + outer * 1000
            snapshots, mean, std, curve = train_controller(
                features, labels, train_indices, effective_seed)
            candidates = []
            for epoch in EPOCHS:
                probabilities = controller_probabilities(
                    features, inner_indices, snapshots[epoch], mean, std,
                    effective_seed)
                selection, grid = search_decision(labels[inner_indices], probabilities)
                candidates.append({"epoch": epoch, "selection": selection,
                                   "decision_grid": grid,
                                   "probabilities": probabilities})
            selected = select_snapshot(candidates)
            state = snapshots[selected["epoch"]]
            outer_probability = controller_probabilities(
                features, outer_indices, state, mean, std, effective_seed)
            outer_metric, outer_score, outer_prediction = decision_metric(
                labels[outer_indices], outer_probability,
                selected["selection"]["threshold"],
                selected["selection"]["risk_penalty"])
            inner_probabilities.append(selected["probabilities"])
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
                "training_curve": curve, "mean": mean, "std": std,
                "state": state,
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
        artifacts[str(outer)] = {
            "inner_fold": inner, "train_folds": train_folds,
            "ensemble_decision": ensemble_selection, "seeds": seed_artifacts,
        }
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
    predictions = [row["prediction"] for row in rows]
    scores = [row["decision_score"] for row in rows]
    ensemble = binary_metrics(labels_all, predictions)
    ensemble.update(rank_metrics(labels_all, scores))
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
        "ensemble_balanced_accuracy_at_least_point_75":
            ensemble["balanced_accuracy"] >= .75,
        "ensemble_auroc_at_least_point_80": ensemble["auroc"] >= .80,
        "all_fold_balanced_accuracy_at_least_point_65": all(
            value["balanced_accuracy"] >= .65 for value in fold_metrics.values()),
        "all_seed_balanced_accuracy_at_least_point_70": all(
            value["balanced_accuracy"] >= .70 for value in per_seed.values()),
    }
    rows_path = output / "oof_residual_controller.jsonl"
    with rows_path.open("x") as sink:
        for row in rows:
            sink.write(json.dumps(row) + "\n")
    checkpoint_path = output / "residual_controllers.pt"
    torch.save({"protocol": PROTOCOL, "feature_names": FEATURE_NAMES,
                "seeds": SEEDS, "folds": artifacts}, checkpoint_path)
    result = {
        "protocol": PROTOCOL, "rows": len(rows), "feature_names": list(FEATURE_NAMES),
        "ensemble_oof": ensemble, "per_seed_oof": per_seed,
        "per_fold_ensemble": fold_metrics, "by_kind": by_kind,
        "selection": selections,
        "controller_gate": {"checks": checks, "passed": all(checks.values())},
        "dependencies": dependencies,
        "hyperparameters": {"hidden": HIDDEN, "epochs": list(EPOCHS),
                            "thresholds": list(THRESHOLDS),
                            "risk_penalties": list(RISK_PENALTIES),
                            "learning_rate": LEARNING_RATE,
                            "weight_decay": WEIGHT_DECAY,
                            "gradient_clip": GRADIENT_CLIP},
        "files_sha256": {rows_path.name: sha(rows_path),
                         checkpoint_path.name: sha(checkpoint_path)},
        "resources": {"seconds": time.perf_counter() - started,
                      "gpu": torch.cuda.get_device_name(),
                      "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                      "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30},
        "relation_ids_used_as_training_labels_only": True,
        "model_outcomes_loaded": False, "sealed_benchmarks_loaded": False,
        "next_stage": ("apply_controller_to_natural_memory_episodes"
                       if all(checks.values())
                       else "analyze_controller_constraint_failure"),
        "terminal_status": "completed_natural_semantic_residual_controller",
    }
    write_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-root", required=True)
    parser.add_argument("--embedding-root", required=True)
    parser.add_argument("--projection-root", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(train(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
