"""Train the preregistered write-time V18 causal deletion policy with nested OOF."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import statistics
import time

from tasks.build_v18_causal_deletion_examples import FEATURE_NAMES, PROTOCOL as DATA_PROTOCOL
from tasks.evaluate_mquake_qwen_outcome_v16 import load_jsonl
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "v18_causal_deletion_policy_v1"
SEEDS = (18031, 18037, 18043)
EPOCHS = (25, 50, 100)
THRESHOLDS = (0.4, 0.5, 0.6)
LOSS_WEIGHTS = {"huber": 1.0, "ranking": 0.5, "conservation": 0.25,
                "consistency": 0.25}
HIDDEN = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0


def verify_examples_release(root):
    root = Path(root); manifest = json.loads((root / "data/manifest.json").read_text())
    audit = json.loads((root / "audit.json").read_text())
    checks = (
        manifest.get("protocol") == DATA_PROTOCOL,
        manifest.get("rows") == 9280,
        manifest.get("feature_names") == list(FEATURE_NAMES),
        manifest.get("target_grid") == [0.0, 0.5, 1.0],
        manifest.get("query_features_used") is False,
        manifest.get("current_update_always_retained") is True,
        manifest.get("examples_sha256") == sha(root / "data/examples.jsonl"),
        audit.get("audit_passed") is True,
        audit.get("manifest_sha256") == sha(root / "data/manifest.json"),
        audit.get("examples_sha256") == sha(root / "data/examples.jsonl"),
        (root / "runner_exit_code.txt").read_text().strip() == "0",
        (root / "terminal_status.txt").read_text().strip()
            == "completed_v18_causal_deletion_examples",
    )
    if not all(checks):
        raise ValueError("policy training requires audited causal deletion examples")
    return {"examples_manifest_sha256": sha(root / "data/manifest.json"),
            "examples_audit_sha256": sha(root / "audit.json"),
            "examples_sha256": sha(root / "data/examples.jsonl")}


def make_model(dim, seed):
    import torch
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(dim, HIDDEN), torch.nn.LayerNorm(HIDDEN), torch.nn.GELU(),
        torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.LayerNorm(HIDDEN), torch.nn.GELU(),
        torch.nn.Linear(HIDDEN, 1),
    )
    for layer in model:
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
    return model


def _tensors(rows, device):
    import torch
    x = torch.tensor([row["features"] for row in rows], dtype=torch.float32, device=device)
    y = torch.tensor([row["target"] for row in rows], dtype=torch.float32, device=device)
    conservation = torch.tensor([row["conservation_target"] for row in rows],
                                dtype=torch.bool, device=device)
    groups = {}
    for index, row in enumerate(rows):
        if row["useful_repetition"]:
            groups.setdefault((row["episode_id"], row["component_key_id"]), []).append(index)
    groups = [torch.tensor(value, dtype=torch.long, device=device)
              for value in groups.values() if len(value) > 1]
    return x, y, conservation, groups


def loss_terms(scores, target, conservation, groups):
    import torch
    import torch.nn.functional as F
    huber = F.smooth_l1_loss(scores, target)
    positive = scores[target > .5]; negative = scores[conservation]
    count = max(len(positive), len(negative))
    if len(positive) and len(negative):
        pos = positive.repeat(math.ceil(count / max(1, len(positive))))[:count]
        neg = negative.repeat(math.ceil(count / max(1, len(negative))))[:count]
        ranking = F.relu(.2 - pos + neg).mean()
    else:
        ranking = scores.new_zeros(())
    conservation_loss = scores[conservation].square().mean() if conservation.any() \
        else scores.new_zeros(())
    consistency = torch.stack([scores[index].var(unbiased=False) for index in groups]).mean() \
        if groups else scores.new_zeros(())
    total = (LOSS_WEIGHTS["huber"] * huber + LOSS_WEIGHTS["ranking"] * ranking
             + LOSS_WEIGHTS["conservation"] * conservation_loss
             + LOSS_WEIGHTS["consistency"] * consistency)
    return total, {"huber": huber, "ranking": ranking,
                   "conservation": conservation_loss, "consistency": consistency}


def fit_snapshots(rows, seed, device):
    import torch
    x, y, conservation, groups = _tensors(rows, device)
    mean, std = x.mean(0), x.std(0, unbiased=False).clamp_min(1e-5)
    normalized = (x - mean) / std
    model = make_model(x.shape[1], seed).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    snapshots, curve = {}, []
    for epoch in range(1, max(EPOCHS) + 1):
        optimizer.zero_grad(set_to_none=True)
        scores = model(normalized).squeeze(-1).sigmoid()
        loss, terms = loss_terms(scores, y, conservation, groups)
        loss.backward(); gradient = float(torch.nn.utils.clip_grad_norm_(
            model.parameters(), GRADIENT_CLIP))
        optimizer.step()
        curve.append({"epoch": epoch, "loss": float(loss.detach()),
                      "gradient_norm": gradient,
                      **{name: float(value.detach()) for name, value in terms.items()}})
        if epoch in EPOCHS:
            snapshots[epoch] = {"state": {name: value.detach().cpu().clone()
                                          for name, value in model.state_dict().items()},
                                "mean": mean.detach().cpu(), "std": std.detach().cpu(),
                                "seed": seed, "epoch": epoch, "dim": x.shape[1]}
    return snapshots, curve


def predict(artifact, rows, device):
    import torch
    model = make_model(artifact["dim"], artifact["seed"])
    model.load_state_dict(artifact["state"]); model.to(device); model.eval()
    x = torch.tensor([row["features"] for row in rows], dtype=torch.float32, device=device)
    mean, std = artifact["mean"].to(device), artifact["std"].to(device)
    with torch.no_grad():
        result = model((x - mean) / std).squeeze(-1).sigmoid().cpu().tolist()
    model.cpu()
    return result


def selection_metrics(rows, scores, threshold):
    if len(rows) != len(scores):
        raise ValueError("score/example length mismatch")
    decisions = [score >= threshold for score in scores]
    positives = [row["target"] > .5 for row in rows]
    conserve = [bool(row["conservation_target"]) for row in rows]
    true_positive = sum(decision and positive for decision, positive in zip(decisions, positives))
    positive_count = sum(positives)
    conserved = sum((not decision) and flag for decision, flag in zip(decisions, conserve))
    conserve_count = sum(conserve)
    recall = true_positive / positive_count if positive_count else 1.0
    specificity = conserved / conserve_count if conserve_count else 1.0
    action_accuracy = statistics.fmean(
        decision == positive for decision, positive in zip(decisions, positives))
    mse = statistics.fmean((score - row["target"]) ** 2
                           for score, row in zip(scores, rows))
    return {"positive_recall": recall, "conservation_specificity": specificity,
            "balanced_accuracy": .5 * (recall + specificity),
            "action_accuracy": action_accuracy, "mse": mse,
            "delete_rate": statistics.fmean(decisions), "rows": len(rows),
            "positive_rows": positive_count, "conservation_rows": conserve_count}


def select_hyperparameters(grid):
    candidates = []
    for epoch in EPOCHS:
        for threshold in THRESHOLDS:
            metric = grid[str(epoch)][str(threshold)]
            satisfied = (metric["positive_recall"] >= .80
                         and metric["conservation_specificity"] >= .95)
            candidates.append((int(satisfied), min(metric["positive_recall"],
                                                   metric["conservation_specificity"]),
                               metric["balanced_accuracy"], metric["action_accuracy"],
                               -metric["mse"], -epoch, -abs(threshold - .5),
                               epoch, threshold))
    selected = max(candidates)
    return {"epoch": selected[-2], "threshold": selected[-1],
            "constraints_satisfied": bool(selected[0]),
            "metrics": grid[str(selected[-2])][str(selected[-1])]}


def policy_states(rows, scores, threshold):
    grouped = {}
    for row, score in zip(rows, scores):
        entry = grouped.setdefault(row["episode_id"], {
            "episode_id": row["episode_id"], "subject": row["subject"],
            "template_group": row["template_group"], "fold": row["fold"],
            "current_session_id": row["current_session_id"], "components": []})
        entry["components"].append({"session_id": row["component_session_id"],
                                    "score": score, "delete": int(score >= threshold),
                                    "target": row["target"]})
    output = []
    for value in grouped.values():
        retained = [row["session_id"] for row in value["components"] if not row["delete"]]
        retained.append(value["current_session_id"])
        output.append({**value, "retained_session_ids": retained,
                       "deleted_session_ids": [row["session_id"]
                                               for row in value["components"] if row["delete"]]})
    return output


def run(args):
    import torch
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    dependencies = verify_examples_release(args.examples_root)
    rows = load_jsonl(Path(args.examples_root) / "data/examples.jsonl")
    if len(rows) != 9280 or set(row["fold"] for row in rows) != set(range(5)):
        raise ValueError("invalid causal deletion training coverage")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    output.mkdir(parents=True); started = time.perf_counter()
    selections, checkpoints, curves, predictions, inner_score_records = {}, {}, {}, [], []
    for outer in range(5):
        inner = (outer + 1) % 5
        inner_train = [row for row in rows if row["fold"] not in (outer, inner)]
        inner_rows = [row for row in rows if row["fold"] == inner]
        inner_artifacts, inner_curves = {}, {}
        for seed in SEEDS:
            inner_artifacts[str(seed)], inner_curves[str(seed)] = fit_snapshots(
                inner_train, seed + outer * 1000, device)
        grid = {}
        for epoch in EPOCHS:
            per_seed = [predict(inner_artifacts[str(seed)][epoch], inner_rows, device)
                        for seed in SEEDS]
            ensemble = [statistics.fmean(values) for values in zip(*per_seed)]
            inner_score_records.append({"outer_fold": outer, "inner_fold": inner,
                                        "epoch": epoch,
                                        "example_ids": [row["example_id"] for row in inner_rows],
                                        "ensemble_scores": ensemble})
            grid[str(epoch)] = {str(threshold): selection_metrics(
                inner_rows, ensemble, threshold) for threshold in THRESHOLDS}
        selection = select_hyperparameters(grid)
        selections[str(outer)] = {"inner_fold": inner, **selection, "grid": grid}
        curves[f"outer{outer}_inner"] = inner_curves
        outer_train = [row for row in rows if row["fold"] != outer]
        outer_rows = [row for row in rows if row["fold"] == outer]
        checkpoints[str(outer)] = {}
        seed_states = {}
        for seed in SEEDS:
            artifacts, curve = fit_snapshots(outer_train, seed + outer * 10_000, device)
            artifact = artifacts[selection["epoch"]]
            checkpoints[str(outer)][str(seed)] = artifact
            scores = predict(artifact, outer_rows, device)
            seed_states[str(seed)] = {row["episode_id"]: row for row in policy_states(
                outer_rows, scores, selection["threshold"])}
            curves[f"outer{outer}_seed{seed}"] = curve
        for episode_id in sorted(seed_states[str(SEEDS[0])]):
            first = seed_states[str(SEEDS[0])][episode_id]
            predictions.append({key: first[key] for key in (
                "episode_id", "subject", "template_group", "fold", "current_session_id")}
                | {"selection": {key: selections[str(outer)][key] for key in (
                    "inner_fold", "epoch", "threshold", "constraints_satisfied")},
                   "per_seed": {str(seed): seed_states[str(seed)][episode_id] for seed in SEEDS}})
    prediction_path = output / "oof_policy_states.jsonl"
    prediction_path.write_text("".join(json.dumps(row) + "\n" for row in predictions))
    inner_path = output / "inner_selection_scores.jsonl"
    inner_path.write_text("".join(json.dumps(row) + "\n" for row in inner_score_records))
    checkpoint_path = output / "controllers.pt"
    torch.save({"protocol": PROTOCOL, "feature_names": FEATURE_NAMES,
                "seeds": SEEDS, "selections": selections,
                "checkpoints": checkpoints}, checkpoint_path)
    result = {"protocol": PROTOCOL, "rows": len(rows), "episodes": len(predictions),
              "seeds": list(SEEDS), "epochs": list(EPOCHS),
              "thresholds": list(THRESHOLDS), "feature_names": list(FEATURE_NAMES),
              "hidden": HIDDEN, "loss_weights": LOSS_WEIGHTS,
              "optimizer": {"name": "AdamW", "learning_rate": LEARNING_RATE,
                            "weight_decay": WEIGHT_DECAY,
                            "gradient_clip": GRADIENT_CLIP},
              "selection": selections, "dependencies": dependencies,
              "checkpoint_sha256": sha(checkpoint_path),
              "oof_policy_states_sha256": sha(prediction_path),
              "inner_selection_scores_sha256": sha(inner_path),
              "training_seconds": time.perf_counter() - started, "device": device,
              "gpu_peak_allocated_gib": (torch.cuda.max_memory_allocated() / 2**30
                                         if device == "cuda" else 0.0),
              "gpu_peak_reserved_gib": (torch.cuda.max_memory_reserved() / 2**30
                                        if device == "cuda" else 0.0),
              "query_features_used": False, "current_update_always_retained": True,
              "external_sets_loaded": False, "terminal_status": "trained_oof_deletion_policy"}
    write_json(output / "training.json", {"curves": curves, **result})
    write_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples-root", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
