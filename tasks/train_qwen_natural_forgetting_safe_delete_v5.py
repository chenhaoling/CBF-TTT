"""Train a grouped cross-fit lifecycle safe-delete head on V4 policy states."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import time

from tasks.evaluate_qwen_natural_forgetting_consensus_v4 import (
    PROTOCOL as SOURCE_PROTOCOL,
)
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_qwen_natural_forgetting_barrier_q_v1 import (
    INNER_FOLDS,
    OUTER_FOLDS,
    SEEDS,
    fold_for,
    inner_fold_for,
)
from tasks.train_qwen_natural_forgetting_factor_gate_v2 import (
    BinaryHead,
    ENGINEERED,
    NO_SCHEDULE,
)


PROTOCOL = "qwen_natural_forgetting_safe_delete_v5"
PLAN = "QWEN_NATURAL_FORGETTING_SAFE_DELETE_V5_PLAN.md"
HIDDENS = (0, 16, 32)
LEARNING_RATES = (.001, .005)
WEIGHT_DECAYS = (0., .01)
THRESHOLDS = (.5, .7, .8, .9, .95, .98, .99)
EPOCHS = 300


def configs():
    return [
        {"hidden": hidden, "learning_rate": lr, "weight_decay": decay}
        for hidden in HIDDENS
        for lr in LEARNING_RATES
        for decay in WEIGHT_DECAYS
    ]


def state_id(session_id, step, memory):
    payload = json.dumps(
        [session_id, step, [list(record) for record in memory]],
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def reconstruct_trace(trace):
    """Reconstruct and deduplicate all arm states from one V4 trace."""
    session_id = trace["session_id"]
    outer_fold = int(trace["outer_fold"])
    if outer_fold != fold_for(session_id):
        raise ValueError("V4 outer-fold mismatch")
    unique = {}
    for arm, decisions in trace["decisions"].items():
        memory = []
        occurrences = Counter()
        for expected_step, decision in enumerate(decisions):
            if int(decision["step"]) != expected_step:
                raise ValueError("non-contiguous V4 event steps")
            if int(decision["before_size"]) != len(memory):
                raise ValueError("V4 before_size does not match reconstructed state")
            incoming = decision["incoming_chain_id"]
            incoming_version = occurrences[incoming]
            top = decision["top"]
            if (top is None) != (not memory):
                raise ValueError("V4 top/memory mismatch")
            if top is not None and not 0 <= int(top) < len(memory):
                raise ValueError("V4 top is outside memory")
            safe = int(
                decision["desired_source_action"] == "local"
                and top is not None
                and memory[int(top)][0] == incoming
            )
            full = tuple(float(value) for value in decision["features"])
            if len(full) != len(ENGINEERED):
                raise ValueError("unexpected V4 feature width")
            feature_values = tuple(full[ENGINEERED.index(name)] for name in NO_SCHEDULE)
            key = (session_id, expected_step, tuple(memory))
            canonical = {
                "protocol": PROTOCOL,
                "state_id": state_id(session_id, expected_step, memory),
                "session_id": session_id,
                "outer_fold": outer_fold,
                "event_step": expected_step,
                "memory": [list(record) for record in memory],
                "top": top,
                "incoming_chain_id": incoming,
                "incoming_version": incoming_version,
                "event_kind": decision["event_kind"],
                "desired_source_action": decision["desired_source_action"],
                "features": list(feature_values),
                "safe_delete": safe,
                "origin_arms": [arm],
            }
            if key in unique:
                previous = unique[key]
                comparable = {name: value for name, value in canonical.items()
                              if name != "origin_arms"}
                previous_comparable = {name: value for name, value in previous.items()
                                       if name != "origin_arms"}
                if comparable != previous_comparable:
                    raise ValueError("duplicate V4 state disagrees across behavior arms")
                previous["origin_arms"].append(arm)
            else:
                unique[key] = canonical
            if decision["action"] == "scope_top1_a0":
                if top is None:
                    raise ValueError("V4 deletes from empty memory")
                memory.pop(int(top))
            elif decision["action"] != "retain_all":
                raise ValueError("unexpected V4 action")
            memory.append((incoming, incoming_version))
            occurrences[incoming] += 1
            if int(decision["after_write_size"]) != len(memory):
                raise ValueError("V4 after_write_size does not match transition")
    rows = list(unique.values())
    for row in rows:
        row["origin_arms"].sort()
    rows.sort(key=lambda row: (row["event_step"], row["state_id"]))
    return rows


def load_corpus(source_root):
    root = Path(source_root)
    summary_path = root / "summary.json"
    audit_path = root / "execution_audit.json"
    summary = json.loads(summary_path.read_text())
    audit = json.loads(audit_path.read_text())
    if summary.get("protocol") != SOURCE_PROTOCOL:
        raise ValueError("unexpected V4 source protocol")
    if summary["gate"]["test_authorized"] or summary.get("selected") is not None:
        raise ValueError("V5 expects the registered V4 calibration stop")
    if not audit.get("passed"):
        raise ValueError("V4 execution audit did not pass")
    rows = []
    for shard in range(2):
        path = root / f"decisions_shard{shard}.jsonl"
        if summary["decisions_sha256"][path.name] != sha(path):
            raise ValueError("V4 decision shard hash mismatch")
        for line in path.read_text().splitlines():
            if line:
                rows.extend(reconstruct_trace(json.loads(line)))
    identities = [row["state_id"] for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("state ID collision across sessions")
    rows.sort(key=lambda row: (row["session_id"], row["event_step"], row["state_id"]))
    return rows, {
        "summary": summary_path,
        "audit": audit_path,
        "decisions_shard0": root / "decisions_shard0.jsonl",
        "decisions_shard1": root / "decisions_shard1.jsonl",
    }


def classification_metrics(targets, probabilities, threshold, groups=None):
    actions = [float(value) >= threshold for value in probabilities]
    tp = sum(action and target for action, target in zip(actions, targets))
    fp = sum(action and not target for action, target in zip(actions, targets))
    fn = sum(not action and target for action, target in zip(actions, targets))
    tn = len(targets) - tp - fp - fn
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    result = {
        "rows": len(targets), "true_positives": tp, "false_positives": fp,
        "false_negatives": fn, "true_negatives": tn,
        "precision": precision, "recall": recall,
        "delete_rate": (tp + fp) / max(1, len(targets)),
    }
    if groups is not None:
        result["by_group"] = {}
        for group in sorted(set(groups)):
            indices = [index for index, value in enumerate(groups) if value == group]
            child = classification_metrics(
                [targets[index] for index in indices],
                [probabilities[index] for index in indices], threshold,
            )
            result["by_group"][str(group)] = child
    return result


def fit_head(features, targets, indices, config, seed):
    import torch

    mean = features[indices].mean(0)
    std = features[indices].std(0, unbiased=False).clamp_min(1e-6)
    x = (features[indices] - mean) / std
    model = BinaryHead.create(features.shape[1], config["hidden"], seed)
    positives = targets[indices].sum()
    negatives = len(indices) - positives
    pos_weight = (negatives / positives.clamp_min(1)).clamp(1, 20)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    for _ in range(EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x).squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets[indices], pos_weight=pos_weight,
        )
        loss.backward()
        optimizer.step()
    model.eval()
    return model, mean, std


def predict(model, mean, std, features):
    import torch

    with torch.no_grad():
        return model((features - mean) / std).squeeze(-1).sigmoid()


def inner_search(features, targets, rows, outer):
    import torch

    training = [index for index, row in enumerate(rows) if row["outer_fold"] != outer]
    search = []
    for config_index, config in enumerate(configs()):
        held = {seed: torch.empty(len(training)) for seed in SEEDS}
        position = {index: local for local, index in enumerate(training)}
        for seed in SEEDS:
            for inner in range(INNER_FOLDS):
                validation = [index for index in training
                              if inner_fold_for(rows[index]["session_id"], outer) == inner]
                validation_set = set(validation)
                fit = [index for index in training if index not in validation_set]
                model, mean, std = fit_head(features, targets, fit, config, seed)
                values = predict(model, mean, std, features[validation])
                for local, index in enumerate(validation):
                    held[seed][position[index]] = values[local]
        probabilities = torch.stack([held[seed] for seed in SEEDS]).mean(0).tolist()
        labels = targets[training].tolist()
        groups = [inner_fold_for(rows[index]["session_id"], outer) for index in training]
        for threshold in THRESHOLDS:
            metrics = classification_metrics(labels, probabilities, threshold, groups)
            active_groups = [value for value in metrics["by_group"].values()
                             if value["true_positives"] + value["false_positives"]]
            minimum_group_precision = min(
                (value["precision"] for value in active_groups), default=1.0,
            )
            feasible = (
                metrics["precision"] >= .98
                and metrics["recall"] >= .50
                and metrics["false_positives"] <= 4
                and minimum_group_precision >= .95
            )
            search.append({
                "config_index": config_index, "config": config,
                "threshold": threshold, "metrics": metrics,
                "minimum_active_inner_fold_precision": minimum_group_precision,
                "feasible": feasible,
            })
    feasible = [row for row in search if row["feasible"]]
    if feasible:
        selected = max(feasible, key=lambda row: (
            row["metrics"]["recall"], -row["metrics"]["false_positives"],
            row["metrics"]["precision"], -row["config"]["hidden"],
            row["threshold"], -row["config_index"],
        ))
    else:
        selected = min(search, key=lambda row: (
            row["metrics"]["false_positives"] > 4,
            row["metrics"]["recall"] < .50,
            sum(value["true_positives"] + value["false_positives"] > 0
                and value["precision"] < .95
                for value in row["metrics"]["by_group"].values()),
            -row["metrics"]["recall"], row["metrics"]["false_positives"],
            -row["metrics"]["precision"], row["config"]["hidden"],
            -row["threshold"], row["config_index"],
        ))
    return training, search, selected


def run(args):
    import torch

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    started = time.perf_counter()
    rows, sources = load_corpus(args.source_root)
    corpus_path = output / "safe_delete_corpus.jsonl"
    with corpus_path.open("x") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    features = torch.tensor([row["features"] for row in rows], dtype=torch.float32)
    targets = torch.tensor([row["safe_delete"] for row in rows], dtype=torch.float32)
    feature_path = output / "safe_delete_features.pt"
    torch.save({"protocol": PROTOCOL, "feature_names": NO_SCHEDULE,
                "features": features, "targets": targets}, feature_path)
    artifact = {"protocol": PROTOCOL, "feature_names": NO_SCHEDULE,
                "seeds": SEEDS, "folds": {}}
    oof = [None] * len(rows)
    searches = {}
    for outer in range(OUTER_FOLDS):
        training, search, selected = inner_search(features, targets, rows, outer)
        validation = [index for index, row in enumerate(rows)
                      if row["outer_fold"] == outer]
        checkpoints = {}
        probabilities = []
        for seed in SEEDS:
            model, mean, std = fit_head(
                features, targets, training, selected["config"], seed,
            )
            values = predict(model, mean, std, features[validation])
            probabilities.append(values)
            checkpoints[str(seed)] = {
                "state": model.state_dict(), "mean": mean, "std": std,
            }
        ensemble = torch.stack(probabilities).mean(0).tolist()
        for local, index in enumerate(validation):
            probability = float(ensemble[local])
            oof[index] = {
                "protocol": PROTOCOL, "state_id": rows[index]["state_id"],
                "session_id": rows[index]["session_id"], "outer_fold": outer,
                "safe_delete": rows[index]["safe_delete"],
                "probability": probability,
                "action": "scope_top1_a0" if probability >= selected["threshold"]
                          else "retain_all",
            }
        artifact["folds"][str(outer)] = {
            "config": selected["config"], "threshold": selected["threshold"],
            "inner_feasible": selected["feasible"],
            "inner_selected_metrics": selected["metrics"],
            "train_sessions": sorted({rows[index]["session_id"] for index in training}),
            "validation_sessions": sorted({rows[index]["session_id"] for index in validation}),
            "checkpoints": checkpoints,
        }
        searches[str(outer)] = search
    if any(row is None for row in oof):
        raise ValueError("incomplete V5 OOF predictions")
    checkpoint_path = output / "safe_delete_crossfit.pt"
    torch.save(artifact, checkpoint_path)
    oof_path = output / "oof_predictions.jsonl"
    with oof_path.open("x") as sink:
        for row in oof:
            sink.write(json.dumps(row, sort_keys=True) + "\n")
    search_path = output / "inner_search.json"
    write_json(search_path, searches)
    labels = [row["safe_delete"] for row in rows]
    probabilities = [row["probability"] for row in oof]
    actions = [row["action"] == "scope_top1_a0" for row in oof]
    # OOF thresholds are fold-specific; use actions directly for combined metrics.
    tp = sum(action and target for action, target in zip(actions, labels))
    fp = sum(action and not target for action, target in zip(actions, labels))
    fn = sum(not action and target for action, target in zip(actions, labels))
    tn = len(labels) - tp - fp - fn
    metrics = {
        "rows": len(labels), "true_positives": tp, "false_positives": fp,
        "false_negatives": fn, "true_negatives": tn,
        "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
        "delete_rate": (tp + fp) / len(labels), "by_outer_fold": {},
    }
    for outer in range(OUTER_FOLDS):
        indices = [index for index, row in enumerate(rows) if row["outer_fold"] == outer]
        fold_actions = [actions[index] for index in indices]
        fold_targets = [labels[index] for index in indices]
        ftp = sum(a and y for a, y in zip(fold_actions, fold_targets))
        ffp = sum(a and not y for a, y in zip(fold_actions, fold_targets))
        ffn = sum(not a and y for a, y in zip(fold_actions, fold_targets))
        metrics["by_outer_fold"][str(outer)] = {
            "rows": len(indices), "true_positives": ftp, "false_positives": ffp,
            "false_negatives": ffn, "precision": ftp / max(1, ftp + ffp),
            "recall": ftp / max(1, ftp + ffn),
        }
    active_folds = [value for value in metrics["by_outer_fold"].values()
                    if value["true_positives"] + value["false_positives"]]
    checks = {
        "precision_at_least_point_98": metrics["precision"] >= .98,
        "recall_at_least_point_50": metrics["recall"] >= .50,
        "false_positive_deletes_at_most_four": fp <= 4,
        "every_active_outer_fold_precision_at_least_point_95": min(
            (value["precision"] for value in active_folds), default=1.0) >= .95,
        "delete_rate_between_point_05_and_point_50": .05 <= metrics["delete_rate"] <= .50,
        "all_sessions_outer_held_out": all(
            not set(value["train_sessions"]) & set(value["validation_sessions"])
            for value in artifact["folds"].values()),
        "privileged_fields_excluded_from_features": tuple(NO_SCHEDULE) == (
            "has_memory", "top_cosine", "cosine_margin", "cosine_mean",
            "cosine_std", "cosine_entropy", "v5_probability",
            "lexical_jaccard", "lexical_containment", "length_ratio",
        ),
        "development_test_external_sealed": True,
    }
    summary = {
        "protocol": PROTOCOL, "source_protocol": SOURCE_PROTOCOL,
        "stage": "offline_crossfit", "rows": len(rows),
        "sessions": len({row["session_id"] for row in rows}),
        "raw_decisions": sum(len(trace["decisions"]) * len(next(iter(trace["decisions"].values())))
                             for shard in range(2)
                             for trace in (json.loads(line) for line in
                                 (Path(args.source_root) / f"decisions_shard{shard}.jsonl")
                                 .read_text().splitlines() if line)),
        "label_distribution": dict(sorted(Counter(labels).items())),
        "feature_names": NO_SCHEDULE, "metrics": metrics,
        "fold_selections": {fold: {
            "config": value["config"], "threshold": value["threshold"],
            "inner_feasible": value["inner_feasible"],
            "inner_selected_metrics": value["inner_selected_metrics"],
        } for fold, value in artifact["folds"].items()},
        "gate": {"checks": checks,
                 "offline_crossfit_passed": all(checks.values()),
                 "online_calibration_authorized": all(checks.values())},
        "hashes": {
            "plan": sha(PLAN), "corpus": sha(corpus_path),
            "features": sha(feature_path), "checkpoint": sha(checkpoint_path),
            "oof": sha(oof_path), "inner_search": sha(search_path),
            **{f"source_{name}": sha(path) for name, path in sources.items()},
        },
        "seconds": time.perf_counter() - started,
        "calibration_loaded": True, "development_loaded": False,
        "test_loaded": False, "external_benchmarks_loaded": False,
    }
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
