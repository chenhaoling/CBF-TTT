"""Train an entity-key augmented lifecycle safe-delete controller V7."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import time

from tasks.build_qwen_natural_causal_memory_v1 import load
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_qwen_natural_forgetting_barrier_q_v1 import (
    INNER_FOLDS,
    OUTER_FOLDS,
    SEEDS,
    fold_for,
    inner_fold_for,
)
from tasks.train_qwen_natural_forgetting_factor_gate_v2 import NO_SCHEDULE
from tasks.train_qwen_natural_forgetting_safe_delete_dagger_v6 import (
    PROTOCOL as SOURCE_PROTOCOL,
    oof_metrics,
)
from tasks.train_qwen_natural_forgetting_safe_delete_v5 import (
    THRESHOLDS,
    classification_metrics,
    configs,
    fit_head,
)


PROTOCOL = "qwen_natural_forgetting_entity_key_v7"
PLAN = "QWEN_NATURAL_FORGETTING_ENTITY_KEY_V7_PLAN.md"
TOKEN_PATTERN = re.compile(r"[\w]+(?:['’\-][\w]+)*", re.UNICODE)
ANCHOR_STOPWORDS = frozenset({
    "a", "an", "and", "are", "at", "by", "did", "do", "does", "for",
    "from", "how", "in", "is", "name", "of", "on", "or", "the", "to",
    "was", "were", "what", "when", "where", "which", "who", "whom",
    "whose", "why",
})
ENTITY_FEATURES = (
    "entity_anchor_both_nonempty", "entity_anchor_jaccard",
    "entity_anchor_containment", "entity_anchor_exact",
    "entity_anchor_count_ratio",
)
FEATURE_NAMES = (*NO_SCHEDULE, *ENTITY_FEATURES)


def entity_anchors(text):
    values = set()
    for token in TOKEN_PATTERN.findall(text):
        normalized = token.casefold()
        if normalized in ANCHOR_STOPWORDS:
            continue
        if any(character.isdigit() for character in token) or \
                (token and token[0].isupper()):
            values.add(normalized)
    return values


def entity_features(left, right):
    if not left or not right:
        return (0., 0., 0., 0., 0.), False, (), ()
    a, b = entity_anchors(left), entity_anchors(right)
    intersection = a & b
    union = a | b
    both = bool(a and b)
    jaccard = len(intersection) / max(1, len(union))
    containment = len(intersection) / max(1, min(len(a), len(b)))
    exact = float(both and a == b)
    ratio = min(len(a), len(b)) / max(1, max(len(a), len(b)))
    barrier = both and containment == 1.0
    return (float(both), jaccard, containment, exact, ratio), barrier, \
        tuple(sorted(a)), tuple(sorted(b))


def augment_corpus(source_root, data_root):
    source = Path(source_root)
    source_summary = json.loads((source / "result" / "summary.json").read_text())
    source_audit = json.loads((source / "execution_audit.json").read_text())
    if source_summary.get("protocol") != SOURCE_PROTOCOL or not source_audit.get("passed"):
        raise ValueError("V6 source is not audited")
    if source_summary["gate"]["online_calibration_authorized"]:
        raise ValueError("entity-key branch expects the V6 offline stop")
    if (source / "terminal_status.txt").read_text().strip() != \
            "completed_qwen_natural_forgetting_safe_delete_dagger_v6_offline_stop":
        raise ValueError("unexpected V6 terminal status")
    rows = [json.loads(line) for line in
            (source / "result" / "safe_delete_dagger_corpus.jsonl")
            .read_text().splitlines() if line]
    _, sessions = load(data_root, "calibration")
    sessions_by_id = {session["id"]: session for session in sessions}
    output = []
    for row in rows:
        session = sessions_by_id[row["session_id"]]
        chains = {chain["id"]: chain for chain in session["chains"]}
        incoming = chains[row["incoming_chain_id"]]["versions"][row["incoming_version"]][
            "prompt"]
        if row["top"] is None:
            top_prompt = None
        else:
            chain_id, version = row["memory"][int(row["top"])]
            top_prompt = chains[chain_id]["versions"][int(version)]["prompt"]
        values, barrier, incoming_anchors, top_anchors = entity_features(incoming, top_prompt)
        output.append({
            **row, "protocol": PROTOCOL,
            "features": [*row["features"], *values],
            "entity_anchor_barrier": barrier,
            "incoming_anchor_tokens": list(incoming_anchors),
            "top_anchor_tokens": list(top_anchors),
            "incoming_prompt_sha256": sha_text(incoming),
            "top_prompt_sha256": None if top_prompt is None else sha_text(top_prompt),
        })
    output.sort(key=lambda row: (row["session_id"], row["event_step"], row["state_id"]))
    return output


def sha_text(value):
    import hashlib
    return hashlib.sha256(value.encode()).hexdigest()


def entity_inner_search(features, targets, rows, outer):
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
                with torch.no_grad():
                    values = model(((features[validation] - mean) / std)) \
                        .squeeze(-1).sigmoid()
                for local, index in enumerate(validation):
                    held[seed][position[index]] = values[local]
        probabilities = torch.stack([held[seed] for seed in SEEDS]).mean(0).tolist()
        labels = targets[training].tolist()
        groups = [inner_fold_for(rows[index]["session_id"], outer) for index in training]
        barriers = [rows[index]["entity_anchor_barrier"] for index in training]
        for threshold in THRESHOLDS:
            for use_barrier in (False, True):
                effective = [probability if not use_barrier or barrier else 0.0
                             for probability, barrier in zip(probabilities, barriers)]
                metrics = classification_metrics(labels, effective, threshold, groups)
                active_groups = [value for value in metrics["by_group"].values()
                                 if value["true_positives"] + value["false_positives"]]
                minimum = min((value["precision"] for value in active_groups), default=1.0)
                feasible = metrics["precision"] >= .98 and metrics["recall"] >= .50 \
                    and metrics["false_positives"] <= 4 and minimum >= .95
                search.append({
                    "config_index": config_index, "config": config,
                    "threshold": threshold, "use_entity_anchor_barrier": use_barrier,
                    "metrics": metrics, "minimum_active_inner_fold_precision": minimum,
                    "feasible": feasible,
                })
    feasible = [row for row in search if row["feasible"]]
    if feasible:
        selected = max(feasible, key=lambda row: (
            row["metrics"]["recall"], -row["metrics"]["false_positives"],
            row["metrics"]["precision"], row["use_entity_anchor_barrier"],
            -row["config"]["hidden"], row["threshold"], -row["config_index"],
        ))
    else:
        selected = min(search, key=lambda row: (
            row["metrics"]["false_positives"] > 4,
            row["metrics"]["recall"] < .50,
            sum(value["true_positives"] + value["false_positives"] > 0
                and value["precision"] < .95
                for value in row["metrics"]["by_group"].values()),
            -row["metrics"]["recall"], row["metrics"]["false_positives"],
            -row["metrics"]["precision"], not row["use_entity_anchor_barrier"],
            row["config"]["hidden"], -row["threshold"], row["config_index"],
        ))
    return training, search, selected


def run(args):
    import torch

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    started = time.perf_counter()
    rows = augment_corpus(args.source_root, args.data_root)
    corpus_path = output / "entity_key_corpus.jsonl"
    with corpus_path.open("x") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    features = torch.tensor([row["features"] for row in rows], dtype=torch.float32)
    targets = torch.tensor([row["safe_delete"] for row in rows], dtype=torch.float32)
    feature_path = output / "entity_key_features.pt"
    torch.save({"protocol": PROTOCOL, "feature_names": FEATURE_NAMES,
                "features": features, "targets": targets}, feature_path)
    artifact = {"protocol": PROTOCOL, "feature_names": FEATURE_NAMES,
                "seeds": SEEDS, "folds": {}}
    oof = [None] * len(rows)
    searches = {}
    for outer in range(OUTER_FOLDS):
        training, search, selected = entity_inner_search(features, targets, rows, outer)
        validation = [index for index, row in enumerate(rows)
                      if row["outer_fold"] == outer]
        predictions, checkpoints = [], {}
        for seed in SEEDS:
            model, mean, std = fit_head(features, targets, training,
                                        selected["config"], seed)
            with torch.no_grad():
                values = model(((features[validation] - mean) / std)) \
                    .squeeze(-1).sigmoid()
            predictions.append(values)
            checkpoints[str(seed)] = {"state": model.state_dict(), "mean": mean, "std": std}
        ensemble = torch.stack(predictions).mean(0).tolist()
        for local, index in enumerate(validation):
            probability = float(ensemble[local])
            barrier = rows[index]["entity_anchor_barrier"]
            delete = probability >= selected["threshold"] and (
                not selected["use_entity_anchor_barrier"] or barrier)
            oof[index] = {
                "protocol": PROTOCOL, "state_id": rows[index]["state_id"],
                "session_id": rows[index]["session_id"], "outer_fold": outer,
                "safe_delete": rows[index]["safe_delete"], "probability": probability,
                "entity_anchor_barrier": barrier,
                "action": "scope_top1_a0" if delete else "retain_all",
            }
        artifact["folds"][str(outer)] = {
            "config": selected["config"], "threshold": selected["threshold"],
            "use_entity_anchor_barrier": selected["use_entity_anchor_barrier"],
            "inner_feasible": selected["feasible"],
            "inner_selected_metrics": selected["metrics"],
            "train_sessions": sorted({rows[index]["session_id"] for index in training}),
            "validation_sessions": sorted({rows[index]["session_id"] for index in validation}),
            "checkpoints": checkpoints,
        }
        searches[str(outer)] = search
    checkpoint_path = output / "entity_key_crossfit.pt"
    torch.save(artifact, checkpoint_path)
    oof_path = output / "oof_predictions.jsonl"
    with oof_path.open("x") as sink:
        for row in oof:
            sink.write(json.dumps(row, sort_keys=True) + "\n")
    search_path = output / "inner_search.json"
    write_json(search_path, searches)
    metrics = oof_metrics(rows, oof)
    active = [value for value in metrics["by_outer_fold"].values()
              if value["true_positives"] + value["false_positives"]]
    checks = {
        "precision_at_least_point_98": metrics["precision"] >= .98,
        "recall_at_least_point_50": metrics["recall"] >= .50,
        "false_positive_deletes_at_most_four": metrics["false_positives"] <= 4,
        "every_active_outer_fold_precision_at_least_point_95": min(
            (value["precision"] for value in active), default=1.0) >= .95,
        "delete_rate_between_point_05_and_point_50": .05 <= metrics["delete_rate"] <= .50,
        "all_sessions_outer_held_out": all(
            not set(value["train_sessions"]) & set(value["validation_sessions"])
            for value in artifact["folds"].values()),
        "only_prompt_derived_entity_features_added": tuple(FEATURE_NAMES[:10]) ==
            tuple(NO_SCHEDULE) and tuple(FEATURE_NAMES[10:]) == ENTITY_FEATURES,
        "development_test_external_sealed": True,
    }
    source = Path(args.source_root)
    summary = {
        "protocol": PROTOCOL, "source_protocol": SOURCE_PROTOCOL,
        "stage": "offline_entity_key_crossfit", "rows": len(rows),
        "sessions": len({row["session_id"] for row in rows}),
        "label_distribution": dict(sorted(Counter(row["safe_delete"] for row in rows).items())),
        "anchor_barrier_distribution": dict(sorted(Counter(
            row["entity_anchor_barrier"] for row in rows).items())),
        "feature_names": FEATURE_NAMES, "metrics": metrics,
        "fold_selections": {fold: {
            "config": value["config"], "threshold": value["threshold"],
            "use_entity_anchor_barrier": value["use_entity_anchor_barrier"],
            "inner_feasible": value["inner_feasible"],
            "inner_selected_metrics": value["inner_selected_metrics"],
        } for fold, value in artifact["folds"].items()},
        "gate": {"checks": checks, "offline_crossfit_passed": all(checks.values()),
                 "online_calibration_authorized": all(checks.values())},
        "hashes": {
            "plan": sha(PLAN), "corpus": sha(corpus_path), "features": sha(feature_path),
            "checkpoint": sha(checkpoint_path), "oof": sha(oof_path),
            "inner_search": sha(search_path),
            "source_summary": sha(source / "result" / "summary.json"),
            "source_audit": sha(source / "execution_audit.json"),
            "source_corpus": sha(source / "result" / "safe_delete_dagger_corpus.jsonl"),
            "data_manifest": sha(Path(args.data_root) / "manifest.json"),
        },
        "seconds": time.perf_counter() - started, "calibration_loaded": True,
        "development_loaded": False, "test_loaded": False,
        "external_benchmarks_loaded": False,
    }
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
