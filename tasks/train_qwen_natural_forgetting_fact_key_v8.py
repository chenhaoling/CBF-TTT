"""Train an independent CounterFact fact-key barrier and freeze-transfer it."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import time

from tasks.build_counterfact_zero_shot_forgetting import SOURCE_SHA256, row_hash
from tasks.build_qwen_natural_causal_memory_v1 import load
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_qwen_natural_forgetting_barrier_q_v1 import SEEDS, lexical_features
from tasks.train_qwen_natural_forgetting_entity_key_v7 import (
    PROTOCOL as SOURCE_PROTOCOL,
    TOKEN_PATTERN,
    entity_anchors,
    entity_features,
)
from tasks.train_qwen_natural_forgetting_factor_gate_v2 import BinaryHead
from tasks.train_qwen_natural_forgetting_safe_delete_dagger_v6 import oof_metrics
from tasks.train_qwen_natural_forgetting_safe_delete_v5 import (
    THRESHOLDS,
    configs,
    fit_head,
    predict,
)


PROTOCOL = "qwen_natural_forgetting_fact_key_v8"
PLAN = "QWEN_NATURAL_FORGETTING_FACT_KEY_V8_PLAN.md"
FEATURE_NAMES = (
    "entity_anchor_both_nonempty", "entity_anchor_jaccard",
    "entity_anchor_containment", "entity_anchor_exact",
    "entity_anchor_count_ratio", "full_lexical_jaccard",
    "full_lexical_containment", "full_length_ratio",
    "relation_lexical_jaccard", "relation_lexical_containment",
    "relation_length_ratio",
)


def subject_fold(subject):
    return int(hashlib.sha256(
        f"fact-key-v8-subject|{subject.casefold()}".encode()).hexdigest(), 16) % 5


def clean_prompt(row):
    rewrite = row.get("requested_rewrite", {})
    subject = str(rewrite.get("subject", "")).strip()
    template = rewrite.get("prompt", "")
    try:
        return template.format(subject).strip()
    except (IndexError, KeyError, ValueError):
        return ""


def valid_row(row):
    rewrite = row.get("requested_rewrite", {})
    return bool(str(rewrite.get("subject", "")).strip()
                and str(rewrite.get("relation_id", "")).strip()
                and clean_prompt(row)
                and [value for value in row.get("paraphrase_prompts", [])
                     if str(value).strip()])


def stable_rows(rows):
    return sorted(rows, key=lambda row: (
        hashlib.sha256(str(row.get("case_id", row_hash(row))).encode()).hexdigest(),
        row_hash(row),
    ))


def build_examples(source):
    source = Path(source)
    if sha(source) != SOURCE_SHA256:
        raise ValueError("unexpected CounterFact source hash")
    raw = [row for row in json.loads(source.read_text()) if valid_row(row)]
    folds = {fold: [] for fold in range(5)}
    for row in raw:
        folds[subject_fold(row["requested_rewrite"]["subject"])].append(row)
    examples, exclusions = [], Counter()
    for fold, fold_rows in folds.items():
        by_relation, by_subject = defaultdict(list), defaultdict(list)
        for row in fold_rows:
            rewrite = row["requested_rewrite"]
            by_relation[str(rewrite["relation_id"])].append(row)
            by_subject[str(rewrite["subject"]).casefold()].append(row)
        for values in by_relation.values():
            values[:] = stable_rows(values)
        for values in by_subject.values():
            values[:] = stable_rows(values)
        for row in stable_rows(fold_rows):
            rewrite = row["requested_rewrite"]
            subject = str(rewrite["subject"])
            relation = str(rewrite["relation_id"])
            paraphrase = next((str(value).strip() for value in row["paraphrase_prompts"]
                               if str(value).strip()), "")
            relation_negatives = [candidate for candidate in by_relation[relation]
                                  if candidate["requested_rewrite"]["subject"].casefold()
                                  != subject.casefold()]
            entity_negatives = [candidate for candidate in by_subject[subject.casefold()]
                                if str(candidate["requested_rewrite"]["relation_id"])
                                != relation]
            if not relation_negatives:
                exclusions["missing_same_relation_different_subject"] += 1
                continue
            if not entity_negatives:
                exclusions["missing_same_subject_different_relation"] += 1
                continue
            identifier = str(row.get("case_id", row_hash(row)))
            common = {"protocol": PROTOCOL, "fold": fold, "group_id": identifier,
                      "left": paraphrase, "left_subject": subject,
                      "left_relation": relation}
            examples.extend((
                {**common, "pair_type": "positive_same_fact_key", "label": 1,
                 "right": clean_prompt(row), "right_subject": subject,
                 "right_relation": relation},
                {**common, "pair_type": "negative_same_relation_different_subject",
                 "label": 0, "right": clean_prompt(relation_negatives[0]),
                 "right_subject": relation_negatives[0]["requested_rewrite"]["subject"],
                 "right_relation": relation},
                {**common, "pair_type": "negative_same_subject_different_relation",
                 "label": 0, "right": clean_prompt(entity_negatives[0]),
                 "right_subject": subject,
                 "right_relation": entity_negatives[0]["requested_rewrite"]["relation_id"]},
            ))
    examples.sort(key=lambda row: (row["fold"], row["group_id"], row["pair_type"]))
    return examples, dict(sorted(exclusions.items()))


def relation_view(text):
    anchors = entity_anchors(text)
    values = []
    for token in TOKEN_PATTERN.findall(text):
        normalized = token.casefold()
        values.append("<entity>" if normalized in anchors else normalized)
    return " ".join(values)


def pair_features(left, right):
    entity, _, _, _ = entity_features(left, right)
    full = lexical_features(left, right)
    relation = lexical_features(relation_view(left), relation_view(right))
    return (*entity, *full, *relation)


def pair_metrics(examples, probabilities, threshold):
    actions = [float(value) >= threshold for value in probabilities]
    labels = [bool(row["label"]) for row in examples]
    tp = sum(a and y for a, y in zip(actions, labels))
    fp = sum(a and not y for a, y in zip(actions, labels))
    fn = sum(not a and y for a, y in zip(actions, labels))
    tn = len(labels) - tp - fp - fn
    by_type = {}
    for pair_type in sorted({row["pair_type"] for row in examples}):
        indices = [index for index, row in enumerate(examples)
                   if row["pair_type"] == pair_type]
        accepted = sum(actions[index] for index in indices)
        positives = sum(labels[index] for index in indices)
        by_type[pair_type] = {
            "examples": len(indices), "accepted": accepted,
            "accuracy": sum(actions[index] == labels[index] for index in indices)
                        / max(1, len(indices)),
            "specificity": None if positives else 1 - accepted / max(1, len(indices)),
        }
    return {"examples": len(examples), "true_positives": tp, "false_positives": fp,
            "false_negatives": fn, "true_negatives": tn,
            "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
            "accuracy": (tp + tn) / max(1, len(labels)), "by_type": by_type}


def feasible_metrics(metrics):
    return metrics["precision"] >= .99 and metrics["recall"] >= .90 and all(
        value["specificity"] is None or value["specificity"] >= .98
        for value in metrics["by_type"].values())


def train_independent(args):
    import torch

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    started = time.perf_counter()
    examples, exclusions = build_examples(args.counterfact_source)
    corpus_path = output / "counterfact_fact_key_pairs.jsonl"
    with corpus_path.open("x") as sink:
        for row in examples:
            sink.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    features = torch.tensor([pair_features(row["left"], row["right"])
                             for row in examples], dtype=torch.float32)
    targets = torch.tensor([row["label"] for row in examples], dtype=torch.float32)
    feature_path = output / "counterfact_fact_key_features.pt"
    torch.save({"protocol": PROTOCOL, "feature_names": FEATURE_NAMES,
                "features": features, "targets": targets}, feature_path)
    train_indices = [index for index, row in enumerate(examples) if row["fold"] <= 2]
    tune_indices = [index for index, row in enumerate(examples) if row["fold"] == 3]
    validation_indices = [index for index, row in enumerate(examples) if row["fold"] == 4]
    search = []
    for config_index, config in enumerate(configs()):
        probabilities = []
        artifacts = []
        for seed in SEEDS:
            model, mean, std = fit_head(features, targets, train_indices, config, seed)
            probabilities.append(predict(model, mean, std, features[tune_indices]))
            artifacts.append((model, mean, std))
        ensemble = torch.stack(probabilities).mean(0).tolist()
        tune_examples = [examples[index] for index in tune_indices]
        for threshold in THRESHOLDS:
            metrics = pair_metrics(tune_examples, ensemble, threshold)
            search.append({"config_index": config_index, "config": config,
                           "threshold": threshold, "metrics": metrics,
                           "feasible": feasible_metrics(metrics)})
    feasible = [row for row in search if row["feasible"]]
    if feasible:
        selected = max(feasible, key=lambda row: (
            row["metrics"]["recall"], row["metrics"]["precision"],
            min(value["specificity"] for value in row["metrics"]["by_type"].values()
                if value["specificity"] is not None),
            -row["config"]["hidden"], row["threshold"], -row["config_index"],
        ))
    else:
        selected = max(search, key=lambda row: (
            feasible_metrics(row["metrics"]), row["metrics"]["accuracy"],
            row["metrics"]["precision"], row["metrics"]["recall"],
            -row["config"]["hidden"], row["threshold"], -row["config_index"],
        ))
    final_train = [index for index, row in enumerate(examples) if row["fold"] <= 3]
    checkpoint = {"protocol": PROTOCOL, "feature_names": FEATURE_NAMES,
                  "config": selected["config"], "threshold": selected["threshold"],
                  "seeds": SEEDS, "heads": {}}
    validation_probabilities = []
    for seed in SEEDS:
        model, mean, std = fit_head(features, targets, final_train,
                                    selected["config"], seed)
        validation_probabilities.append(predict(
            model, mean, std, features[validation_indices]))
        checkpoint["heads"][str(seed)] = {
            "state": model.state_dict(), "mean": mean, "std": std,
        }
    validation_probability = torch.stack(validation_probabilities).mean(0).tolist()
    validation_examples = [examples[index] for index in validation_indices]
    validation_metrics = pair_metrics(
        validation_examples, validation_probability, selected["threshold"])
    checkpoint_path = output / "fact_key_barrier.pt"
    torch.save(checkpoint, checkpoint_path)
    tune_path = output / "tune_search.json"
    write_json(tune_path, search)
    validation_path = output / "validation_predictions.jsonl"
    with validation_path.open("x") as sink:
        for row, probability in zip(validation_examples, validation_probability):
            sink.write(json.dumps({"group_id": row["group_id"], "fold": 4,
                "pair_type": row["pair_type"], "label": row["label"],
                "probability": probability,
                "prediction": int(probability >= selected["threshold"])},
                sort_keys=True) + "\n")
    fold_subjects = {str(fold): sorted({row["left_subject"].casefold()
                                       for row in examples if row["fold"] == fold})
                     for fold in range(5)}
    counts_ok = all(validation_metrics["by_type"].get(name, {}).get("examples", 0) >= 100
                    for name in ("positive_same_fact_key",
                                 "negative_same_relation_different_subject",
                                 "negative_same_subject_different_relation"))
    disjoint = all(not set(fold_subjects[str(left)]) & set(fold_subjects[str(right)])
                   for left in range(5) for right in range(left + 1, 5))
    checks = {
        "validation_precision_at_least_point_99": validation_metrics["precision"] >= .99,
        "validation_recall_at_least_point_90": validation_metrics["recall"] >= .90,
        "both_negative_specificities_at_least_point_98": all(
            value["specificity"] is None or value["specificity"] >= .98
            for value in validation_metrics["by_type"].values()),
        "at_least_100_validation_examples_per_type": counts_ok,
        "subject_folds_disjoint": disjoint,
        "tune_selected_without_validation": True,
        "zsre_sources_not_loaded": True,
    }
    summary = {
        "protocol": PROTOCOL, "stage": "independent_counterfact_validation",
        "pairs": len(examples), "excluded": exclusions,
        "pair_distribution": dict(sorted(Counter(row["pair_type"] for row in examples).items())),
        "fold_distribution": dict(sorted(Counter(row["fold"] for row in examples).items())),
        "fold_subject_hashes": {fold: hashlib.sha256(
            json.dumps(values).encode()).hexdigest() for fold, values in fold_subjects.items()},
        "feature_names": FEATURE_NAMES,
        "selection": selected, "validation": validation_metrics,
        "gate": {"checks": checks, "independent_validation_passed": all(checks.values()),
                 "zsre_transfer_authorized": all(checks.values())},
        "hashes": {"plan": sha(PLAN), "counterfact_source": sha(args.counterfact_source),
                   "corpus": sha(corpus_path), "features": sha(feature_path),
                   "checkpoint": sha(checkpoint_path), "tune_search": sha(tune_path),
                   "validation_predictions": sha(validation_path)},
        "seconds": time.perf_counter() - started, "counterfact_loaded": True,
        "zsre_calibration_loaded": False, "zsre_test_loaded": False,
        "external_benchmarks_loaded": False,
    }
    write_json(output / "summary.json", summary)
    return summary


class FactKeyBarrier:
    def __init__(self, checkpoint):
        self.threshold = float(checkpoint["threshold"])
        self.heads = []
        for seed in SEEDS:
            row = checkpoint["heads"][str(seed)]
            model = BinaryHead.create(len(FEATURE_NAMES),
                                      checkpoint["config"]["hidden"], seed)
            model.load_state_dict(row["state"]); model.eval()
            self.heads.append((model, row["mean"], row["std"]))

    def probability(self, values):
        import torch
        x = torch.tensor(values, dtype=torch.float32)
        with torch.no_grad():
            return float(torch.stack([model(((x - mean) / std).unsqueeze(0))
                        .squeeze().sigmoid() for model, mean, std in self.heads]).mean())


def v7_prompt_pairs(v7_root, data_root):
    rows = [json.loads(line) for line in
            (Path(v7_root) / "result" / "entity_key_corpus.jsonl")
            .read_text().splitlines() if line]
    _, sessions = load(data_root, "calibration")
    table = {session["id"]: session for session in sessions}
    output = []
    for row in rows:
        session = table[row["session_id"]]
        chains = {chain["id"]: chain for chain in session["chains"]}
        incoming = chains[row["incoming_chain_id"]]["versions"][row["incoming_version"]][
            "prompt"]
        top = None
        if row["top"] is not None:
            chain_id, version = row["memory"][int(row["top"])]
            top = chains[chain_id]["versions"][int(version)]["prompt"]
        output.append((row, incoming, top))
    return output


def transfer(args):
    import torch

    independent = Path(args.independent_root)
    summary = json.loads((independent / "summary.json").read_text())
    audit = json.loads((independent / "execution_audit.json").read_text())
    if not summary["gate"]["zsre_transfer_authorized"] or not audit.get("passed"):
        raise ValueError("independent fact-key barrier not released")
    v7 = Path(args.v7_root)
    v7_summary = json.loads((v7 / "result" / "summary.json").read_text())
    v7_audit = json.loads((v7 / "execution_audit.json").read_text())
    if v7_summary.get("protocol") != SOURCE_PROTOCOL or not v7_audit.get("passed") \
            or v7_summary["gate"]["online_calibration_authorized"]:
        raise ValueError("V7 source is not the audited offline stop")
    checkpoint_path = independent / "fact_key_barrier.pt"
    if summary["hashes"]["checkpoint"] != sha(checkpoint_path):
        raise ValueError("fact-key checkpoint hash mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    barrier = FactKeyBarrier(checkpoint)
    source_oof = [json.loads(line) for line in
                  (v7 / "result" / "oof_predictions.jsonl").read_text().splitlines()
                  if line]
    pairs = v7_prompt_pairs(v7, args.data_root)
    if [row[0]["state_id"] for row in pairs] != [row["state_id"] for row in source_oof]:
        raise ValueError("V7 prompt/OOF identity mismatch")
    transferred = []
    for (row, incoming, top), source in zip(pairs, source_oof):
        probability = 0.0 if top is None else barrier.probability(pair_features(incoming, top))
        delete = source["action"] == "scope_top1_a0" and probability >= barrier.threshold
        transferred.append({
            "protocol": PROTOCOL, "state_id": row["state_id"],
            "session_id": row["session_id"], "outer_fold": row["outer_fold"],
            "safe_delete": row["safe_delete"], "v7_action": source["action"],
            "fact_key_probability": probability,
            "fact_key_threshold": barrier.threshold,
            "action": "scope_top1_a0" if delete else "retain_all",
        })
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    rows_path = output / "transfer_predictions.jsonl"
    with rows_path.open("x") as sink:
        for row in transferred:
            sink.write(json.dumps(row, sort_keys=True) + "\n")
    source_rows = [row for row, _, _ in pairs]
    metrics = oof_metrics(source_rows, transferred)
    active = [value for value in metrics["by_outer_fold"].values()
              if value["true_positives"] + value["false_positives"]]
    checks = {
        "precision_at_least_point_98": metrics["precision"] >= .98,
        "recall_at_least_point_50": metrics["recall"] >= .50,
        "false_positive_deletes_at_most_four": metrics["false_positives"] <= 4,
        "every_active_outer_fold_precision_at_least_point_95": min(
            (value["precision"] for value in active), default=1.0) >= .95,
        "delete_rate_between_point_05_and_point_50": .05 <= metrics["delete_rate"] <= .50,
        "independent_counterfact_validation_passed": True,
        "zero_zsre_parameter_selection": True,
        "development_test_external_sealed": True,
    }
    transfer_summary = {
        "protocol": PROTOCOL, "stage": "frozen_zsre_v7_transfer",
        "rows": len(transferred), "metrics": metrics,
        "gate": {"checks": checks, "offline_transfer_passed": all(checks.values()),
                 "online_calibration_authorized": all(checks.values())},
        "hashes": {"plan": sha(PLAN), "independent_summary": sha(
            independent / "summary.json"), "independent_audit": sha(
            independent / "execution_audit.json"), "checkpoint": sha(checkpoint_path),
            "v7_summary": sha(v7 / "result" / "summary.json"),
            "v7_audit": sha(v7 / "execution_audit.json"),
            "v7_oof": sha(v7 / "result" / "oof_predictions.jsonl"),
            "data_manifest": sha(Path(args.data_root) / "manifest.json"),
            "predictions": sha(rows_path)},
        "counterfact_loaded": True, "zsre_calibration_loaded": True,
        "zsre_test_loaded": False, "external_benchmarks_loaded": False,
    }
    write_json(output / "summary.json", transfer_summary)
    return transfer_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--counterfact-source", required=True)
    train_parser.add_argument("--output", required=True)
    transfer_parser = sub.add_parser("transfer")
    transfer_parser.add_argument("--independent-root", required=True)
    transfer_parser.add_argument("--v7-root", required=True)
    transfer_parser.add_argument("--data-root", required=True)
    transfer_parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = train_independent(args) if args.command == "train" else transfer(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
