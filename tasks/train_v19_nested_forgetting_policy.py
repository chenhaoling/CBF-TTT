"""Train the frozen two-head V19 nested-OOF forgetting-policy fallback."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import statistics
import time

from tasks.build_v19_nested_policy_examples import FEATURE_NAMES, PROTOCOL as DATA_PROTOCOL
from tasks.evaluate_mquake_qwen_outcome_v16 import load_jsonl
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "v19_nested_oof_forgetting_policy_v1"
SEEDS = (19031, 19037, 19043)
EPOCHS = (25, 50, 100)
THRESHOLDS = (0.4, 0.5, 0.6)
RISK_PENALTIES = (0.0, 0.5, 1.0, 2.0)
LOSS_WEIGHTS = {"utility": 1.0, "risk": 0.5, "conservation": 0.25,
                "ranking": 0.25}
HIDDEN = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0


def verify_examples_release(root):
    root = Path(root)
    manifest = json.loads((root / "data/manifest.json").read_text())
    audit = json.loads((root / "audit.json").read_text())
    checks = (
        manifest.get("protocol") == DATA_PROTOCOL,
        manifest.get("rows") == 9280,
        manifest.get("episodes") == 320,
        manifest.get("feature_names") == list(FEATURE_NAMES),
        manifest.get("target_grid") == [0.0, .5, 1.0],
        manifest.get("query_features_used") is False,
        manifest.get("answer_features_used") is False,
        manifest.get("current_update_always_retained") is True,
        manifest.get("examples_sha256") == sha(root / "data/examples.jsonl"),
        audit.get("audit_passed") is True,
        audit.get("manifest_sha256") == sha(root / "data/manifest.json"),
        audit.get("examples_sha256") == sha(root / "data/examples.jsonl"),
        (root / "runner_exit_code.txt").read_text().strip() == "0",
        (root / "terminal_status.txt").read_text().strip()
            == "completed_v19_nested_policy_examples",
    )
    if not all(checks):
        raise ValueError("V19 policy training requires audited fallback examples")
    return {"manifest_sha256": sha(root / "data/manifest.json"),
            "audit_sha256": sha(root / "audit.json"),
            "examples_sha256": sha(root / "data/examples.jsonl")}


def make_model(dim, seed):
    import torch
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(dim, HIDDEN), torch.nn.LayerNorm(HIDDEN), torch.nn.GELU(),
        torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.LayerNorm(HIDDEN), torch.nn.GELU(),
        torch.nn.Linear(HIDDEN, 2),
    )


def pair_indices(rows):
    grouped = {}
    for index, row in enumerate(rows):
        grouped.setdefault(row["episode_id"], []).append(index)
    positive, negative = [], []
    for indices in grouped.values():
        pos = [index for index in indices if rows[index]["target"] > .5]
        neg = [index for index in indices if rows[index]["target"] <= .5]
        for left in pos:
            for right in neg:
                positive.append(left); negative.append(right)
    return positive, negative


def fit_snapshots(rows, seed, device):
    import torch
    import torch.nn.functional as functional
    x = torch.tensor([row["features"] for row in rows], dtype=torch.float32,
                     device=device)
    utility = torch.tensor([row["target"] for row in rows], dtype=torch.float32,
                           device=device)
    risk = torch.tensor([row["locality_risk_target"] for row in rows],
                        dtype=torch.float32, device=device)
    conserve = torch.tensor([row["conservation_target"] for row in rows],
                            dtype=torch.bool, device=device)
    pos, neg = pair_indices(rows)
    pos = torch.tensor(pos, dtype=torch.long, device=device)
    neg = torch.tensor(neg, dtype=torch.long, device=device)
    mean, std = x.mean(0), x.std(0).clamp_min(1e-6)
    model = make_model(x.shape[1], seed).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    snapshots, curve = {}, []
    for epoch in range(1, max(EPOCHS) + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        logits = model((x - mean) / std)
        utility_probability = logits[:, 0].sigmoid()
        utility_loss = functional.smooth_l1_loss(utility_probability, utility)
        risk_loss = functional.binary_cross_entropy_with_logits(logits[:, 1], risk)
        conservation_loss = (functional.binary_cross_entropy_with_logits(
            logits[conserve, 0], torch.zeros_like(logits[conserve, 0]))
            if conserve.any() else logits[:, 0].sum() * 0)
        ranking_loss = (functional.softplus(
            -(logits[pos, 0] - logits[neg, 0])).mean()
            if len(pos) else logits[:, 0].sum() * 0)
        loss = (LOSS_WEIGHTS["utility"] * utility_loss
                + LOSS_WEIGHTS["risk"] * risk_loss
                + LOSS_WEIGHTS["conservation"] * conservation_loss
                + LOSS_WEIGHTS["ranking"] * ranking_loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        optimizer.step()
        curve.append({"epoch": epoch, "loss": float(loss.detach().cpu()),
                      "utility_loss": float(utility_loss.detach().cpu()),
                      "risk_loss": float(risk_loss.detach().cpu()),
                      "conservation_loss": float(conservation_loss.detach().cpu()),
                      "ranking_loss": float(ranking_loss.detach().cpu())})
        if epoch in EPOCHS:
            snapshots[epoch] = {"state": copy.deepcopy(model.state_dict()),
                                "mean": mean.detach().cpu(), "std": std.detach().cpu(),
                                "seed": seed, "epoch": epoch, "dim": x.shape[1]}
    model.cpu()
    return snapshots, curve


def predict(artifact, rows, device):
    import torch
    model = make_model(artifact["dim"], artifact["seed"])
    model.load_state_dict(artifact["state"]); model.to(device); model.eval()
    x = torch.tensor([row["features"] for row in rows], dtype=torch.float32,
                     device=device)
    with torch.no_grad():
        logits = model((x - artifact["mean"].to(device)) / artifact["std"].to(device))
        values = logits.sigmoid().cpu().tolist()
    model.cpu()
    return {"utility": [row[0] for row in values], "risk": [row[1] for row in values]}


def selection_metrics(rows, utility, risk, threshold, penalty):
    if not len(rows) == len(utility) == len(risk):
        raise ValueError("V19 policy score/example length mismatch")
    scores = [benefit - penalty * danger for benefit, danger in zip(utility, risk)]
    decisions = [score >= threshold for score in scores]
    positives = [row["target"] > .5 for row in rows]
    conservation = [bool(row["conservation_target"]) for row in rows]
    risky = [bool(row["locality_risk_target"]) for row in rows]
    recall = (sum(action and label for action, label in zip(decisions, positives))
              / max(1, sum(positives)))
    specificity = (sum((not action) and label for action, label in
                       zip(decisions, conservation)) / max(1, sum(conservation)))
    risk_violation = (sum(action and label for action, label in zip(decisions, risky))
                      / max(1, sum(risky)))
    accuracy = statistics.fmean(action == label for action, label in
                                zip(decisions, positives))
    mse = statistics.fmean((score - row["target"]) ** 2
                           for score, row in zip(scores, rows))
    return {"positive_recall": recall, "conservation_specificity": specificity,
            "risk_violation_rate": risk_violation,
            "balanced_utility_accuracy": .5 * (recall + specificity),
            "action_accuracy": accuracy, "mse": mse,
            "delete_rate": statistics.fmean(decisions), "rows": len(rows),
            "positive_rows": sum(positives), "conservation_rows": sum(conservation),
            "locality_risk_rows": sum(risky)}


def select_hyperparameters(grid):
    candidates = []
    for epoch in EPOCHS:
        for threshold in THRESHOLDS:
            for penalty in RISK_PENALTIES:
                value = grid[str(epoch)][str(threshold)][str(penalty)]
                feasible = (value["positive_recall"] >= .80
                            and value["conservation_specificity"] >= .95
                            and value["risk_violation_rate"] <= .03)
                candidates.append((int(feasible), value["balanced_utility_accuracy"],
                                   value["conservation_specificity"],
                                   -value["risk_violation_rate"], -value["delete_rate"],
                                   -epoch, threshold, penalty, epoch))
    selected = max(candidates)
    epoch, threshold, penalty = selected[-1], selected[-3], selected[-2]
    return {"epoch": epoch, "threshold": threshold, "risk_penalty": penalty,
            "constraints_satisfied": bool(selected[0]),
            "metrics": grid[str(epoch)][str(threshold)][str(penalty)]}


def policy_states(rows, predictions, threshold, penalty):
    grouped = {}
    for row, utility, risk in zip(rows, predictions["utility"], predictions["risk"]):
        entry = grouped.setdefault(row["episode_id"], {
            "episode_id": row["episode_id"], "subject": row["subject"],
            "template_group": row["template_group"], "fold": row["fold"],
            "current_session_id": row["current_session_id"], "components": []})
        score = utility - penalty * risk
        entry["components"].append({
            "session_id": row["component_session_id"], "utility": utility,
            "risk": risk, "score": score, "delete": int(score >= threshold),
            "target": row["target"], "locality_risk_target": row["locality_risk_target"]})
    output = []
    for value in grouped.values():
        retained = [row["session_id"] for row in value["components"] if not row["delete"]]
        retained.append(value["current_session_id"])
        output.append({**value, "retained_session_ids": retained,
                       "deleted_session_ids": [row["session_id"]
                                               for row in value["components"]
                                               if row["delete"]]})
    return output


def run(examples_root, output):
    import torch
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    dependencies = verify_examples_release(examples_root)
    rows = load_jsonl(Path(examples_root) / "data/examples.jsonl")
    if len(rows) != 9280 or {row["fold"] for row in rows} != set(range(5)):
        raise ValueError("invalid V19 nested policy coverage")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    output.mkdir(parents=True); started = time.perf_counter()
    selections, checkpoints, curves, inner_records, states = {}, {}, {}, [], []
    for outer in range(5):
        inner = (outer + 1) % 5
        inner_train = [row for row in rows if row["fold"] not in (outer, inner)]
        inner_rows = [row for row in rows if row["fold"] == inner]
        artifacts = {}
        for seed in SEEDS:
            artifacts[str(seed)], curve = fit_snapshots(
                inner_train, seed + outer * 1000, device)
            curves[f"outer{outer}_inner_seed{seed}"] = curve
        grid = {}
        for epoch in EPOCHS:
            per_seed = [predict(artifacts[str(seed)][epoch], inner_rows, device)
                        for seed in SEEDS]
            utility = [statistics.fmean(value) for value in zip(
                *(row["utility"] for row in per_seed))]
            risk = [statistics.fmean(value) for value in zip(
                *(row["risk"] for row in per_seed))]
            inner_records.append({"outer_fold": outer, "inner_fold": inner,
                                  "epoch": epoch,
                                  "example_ids": [row["example_id"] for row in inner_rows],
                                  "utility": utility, "risk": risk})
            grid[str(epoch)] = {str(threshold): {str(penalty): selection_metrics(
                inner_rows, utility, risk, threshold, penalty)
                for penalty in RISK_PENALTIES} for threshold in THRESHOLDS}
        selection = select_hyperparameters(grid)
        selections[str(outer)] = {"inner_fold": inner, **selection, "grid": grid}
        outer_train = [row for row in rows if row["fold"] != outer]
        outer_rows = [row for row in rows if row["fold"] == outer]
        checkpoints[str(outer)] = {}
        seed_states = {}
        for seed in SEEDS:
            trained, curve = fit_snapshots(outer_train, seed + outer * 10_000, device)
            artifact = trained[selection["epoch"]]
            checkpoints[str(outer)][str(seed)] = artifact
            prediction = predict(artifact, outer_rows, device)
            seed_states[str(seed)] = {row["episode_id"]: row for row in policy_states(
                outer_rows, prediction, selection["threshold"],
                selection["risk_penalty"])}
            curves[f"outer{outer}_seed{seed}"] = curve
        for episode_id in sorted(seed_states[str(SEEDS[0])]):
            first = seed_states[str(SEEDS[0])][episode_id]
            states.append({key: first[key] for key in (
                "episode_id", "subject", "template_group", "fold", "current_session_id")}
                | {"selection": {key: selection[key] for key in (
                    "inner_fold", "epoch", "threshold", "risk_penalty",
                    "constraints_satisfied")},
                   "per_seed": {str(seed): seed_states[str(seed)][episode_id]
                                for seed in SEEDS}})
    state_path = output / "oof_policy_states.jsonl"
    state_path.write_text("".join(json.dumps(row) + "\n" for row in states))
    inner_path = output / "inner_selection_scores.jsonl"
    inner_path.write_text("".join(json.dumps(row) + "\n" for row in inner_records))
    checkpoint_path = output / "controllers.pt"
    torch.save({"protocol": PROTOCOL, "feature_names": FEATURE_NAMES,
                "seeds": SEEDS, "selections": selections,
                "checkpoints": checkpoints}, checkpoint_path)
    result = {
        "protocol": PROTOCOL, "rows": len(rows), "episodes": len(states),
        "seeds": list(SEEDS), "epochs": list(EPOCHS),
        "thresholds": list(THRESHOLDS), "risk_penalties": list(RISK_PENALTIES),
        "feature_names": list(FEATURE_NAMES), "hidden": HIDDEN,
        "loss_weights": LOSS_WEIGHTS,
        "optimizer": {"name": "AdamW", "learning_rate": LEARNING_RATE,
                      "weight_decay": WEIGHT_DECAY, "gradient_clip": GRADIENT_CLIP},
        "selection": selections, "dependencies": dependencies,
        "checkpoint_sha256": sha(checkpoint_path),
        "oof_policy_states_sha256": sha(state_path),
        "inner_selection_scores_sha256": sha(inner_path),
        "training_seconds": time.perf_counter() - started, "device": device,
        "gpu_peak_allocated_gib": (torch.cuda.max_memory_allocated() / 2**30
                                   if device == "cuda" else 0.0),
        "gpu_peak_reserved_gib": (torch.cuda.max_memory_reserved() / 2**30
                                  if device == "cuda" else 0.0),
        "query_features_used": False, "answer_features_used": False,
        "current_update_always_retained": True, "external_sets_loaded": False,
        "terminal_status": "trained_v19_nested_oof_forgetting_policy",
    }
    write_json(output / "training.json", {"curves": curves, **result})
    write_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples-root", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
