"""Fit S3 multi-support evidence fusion and conservative calibration."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

from tasks.pack_cbf_event_warmup import sha
from tasks.score_retrieval_evidence_forgetting_s3_pairs import PROTOCOL as PRIOR_PROTOCOL
from tasks.train_retrieval_evidence_forgetting_s3_projection import (
    PROTOCOL as PROJECTION_PROTOCOL,
    ProjectionHead,
    load_embedding_set,
    project_matrix,
    roc_auc,
)


PROTOCOL = "qwen_retrieval_evidence_forgetting_s3_fusion_v1"
DATA_PROTOCOL = "qwen_retrieval_evidence_forgetting_s3"
AUDIT_PROTOCOL = "qwen_retrieval_evidence_forgetting_s3_data_audit_v1"
SEED = 35117
SUPPORT_K = 16
FEATURE_NAMES = (
    "projected_direct_cosine",
    "raw_direct_cosine",
    "s2_prior_score",
    "reciprocal_support_overlap",
    "posterior_dot",
    "negative_posterior_js",
    "minimum_neighbor_margin",
    "negative_max_entropy",
    "support_agreement_product",
    "negative_max_score_variance",
    "cross_support_best",
    "cross_support_median",
    "cross_support_trimmed_mean",
    "type_compatible",
)
DIRECT_FEATURES = {
    "projected_direct_cosine", "raw_direct_cosine", "s2_prior_score", "type_compatible"
}


def load_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def load_prior_scores(root: str | Path, split: str, expected_rows: int) -> tuple[dict[str, float], list[dict]]:
    root = Path(root)
    rows, audits = [], []
    for shard in range(2):
        path = root / f"{split}_shard_{shard}.jsonl"
        audit_path = root / f"{split}_shard_{shard}.audit.json"
        audit = json.loads(audit_path.read_text())
        if (audit.get("protocol") != PRIOR_PROTOCOL
                or audit.get("split") != split
                or audit.get("shard") != shard
                or audit.get("num_shards") != 2
                or audit.get("dialogre_dev_loaded") is not False
                or audit.get("dialogre_test_loaded") is not False
                or sha(path) != audit.get("output_sha256")):
            raise ValueError(f"invalid S3 prior score shard: {split}/{shard}")
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line)
        audits.append(audit)
    rows.sort(key=lambda row: row["index"])
    if len(rows) != expected_rows or [row["index"] for row in rows] != list(range(expected_rows)):
        raise ValueError(f"incomplete S3 prior scores: {split}")
    if len({row["pair_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate S3 prior pair IDs")
    return {row["pair_id"]: row["relation_score"] for row in rows}, audits


def neighbor_evidence(query_matrix, query_dialogues: list[str], bank_matrix,
                      bank_dialogues: list[str], same_bank: bool = False):
    import numpy as np
    import torch

    bank = torch.as_tensor(bank_matrix, dtype=torch.float32, device="cuda")
    top_indices, top_scores = [], []
    bank_dialogues_array = np.asarray(bank_dialogues)
    for start in range(0, len(query_matrix), 128):
        query = torch.as_tensor(query_matrix[start:start + 128], dtype=torch.float32,
                                device="cuda")
        scores = query @ bank.T
        if same_bank:
            query_dialogue = np.asarray(query_dialogues[start:start + len(query)])
            mask = query_dialogue[:, None] == bank_dialogues_array[None, :]
            scores.masked_fill_(torch.as_tensor(mask, device="cuda"), -torch.inf)
        values, indices = torch.topk(scores, k=SUPPORT_K, dim=1)
        if not bool(torch.isfinite(values).all()):
            raise ValueError("S3 support query has fewer than 16 eligible candidates")
        top_indices.append(indices.cpu().numpy())
        top_scores.append(values.cpu().numpy())
    return np.concatenate(top_indices), np.concatenate(top_scores)


def evidence_statistics(top_indices, top_scores, bank_relations: list[str]):
    import numpy as np

    relation_names = sorted(set(bank_relations))
    relation_id = {name: index for index, name in enumerate(relation_names)}
    labels = np.asarray([relation_id[value] for value in bank_relations])
    posteriors = np.zeros((len(top_indices), len(relation_names)), dtype="float32")
    margins, entropies, agreements, variances = [], [], [], []
    for row_index, (indices, scores) in enumerate(zip(top_indices, top_scores)):
        weights = np.exp((scores - scores.max()) / .10)
        weights /= weights.sum()
        for relation, weight in zip(labels[indices], weights):
            posteriors[row_index, relation] += weight
        sorted_posterior = np.sort(posteriors[row_index])[::-1]
        margins.append(float(sorted_posterior[0] - sorted_posterior[1]))
        nonzero = posteriors[row_index][posteriors[row_index] > 0]
        entropy = -float((nonzero * np.log(nonzero)).sum()) / max(1e-12, math.log(len(relation_names)))
        entropies.append(entropy)
        top_relation = labels[indices[0]]
        agreements.append(float((labels[indices] == top_relation).mean()))
        variances.append(float(scores.var()))
    return {
        "posteriors": posteriors,
        "margins": np.asarray(margins, dtype="float32"),
        "entropies": np.asarray(entropies, dtype="float32"),
        "agreements": np.asarray(agreements, dtype="float32"),
        "variances": np.asarray(variances, dtype="float32"),
    }


def reciprocal_overlap(left: list[int], right: list[int]) -> float:
    right_rank = {value: index + 1 for index, value in enumerate(right)}
    score = sum(2.0 / ((index + 1) + right_rank[value])
                for index, value in enumerate(left) if value in right_rank)
    normalizer = sum(1.0 / (index + 1) for index in range(SUPPORT_K))
    return score / normalizer


def negative_js(left, right) -> float:
    import numpy as np

    midpoint = (left + right) / 2
    left_mask, right_mask = left > 0, right > 0
    divergence = .5 * float((left[left_mask] * np.log(
        left[left_mask] / midpoint[left_mask]
    )).sum())
    divergence += .5 * float((right[right_mask] * np.log(
        right[right_mask] / midpoint[right_mask]
    )).sum())
    return -divergence


def pair_feature_table(pairs: list[dict], query_matrix, raw_query_matrix,
                       query_ids: list[str], evidence: dict, top_indices, bank_matrix,
                       prior_scores: dict[str, float]):
    import numpy as np

    index = {source_id: position for position, source_id in enumerate(query_ids)}
    features, gold = [], []
    for row in pairs:
        left, right = (index[row["left_source_row_sha256"]],
                       index[row["right_source_row_sha256"]])
        left_support, right_support = top_indices[left], top_indices[right]
        cross = (bank_matrix[left_support] @ bank_matrix[right_support].T).reshape(-1)
        ordered = np.sort(cross)
        trim = max(1, int(.10 * len(ordered)))
        trimmed = ordered[trim:-trim] if 2 * trim < len(ordered) else ordered
        feature = (
            float(query_matrix[left] @ query_matrix[right]),
            float(raw_query_matrix[left] @ raw_query_matrix[right]),
            float(prior_scores[row["pair_id"]]),
            reciprocal_overlap(left_support.tolist(), right_support.tolist()),
            float(evidence["posteriors"][left] @ evidence["posteriors"][right]),
            negative_js(evidence["posteriors"][left], evidence["posteriors"][right]),
            float(min(evidence["margins"][left], evidence["margins"][right])),
            -float(max(evidence["entropies"][left], evidence["entropies"][right])),
            float(evidence["agreements"][left] * evidence["agreements"][right]),
            -float(max(evidence["variances"][left], evidence["variances"][right])),
            float(ordered[-1]),
            float(np.median(ordered)),
            float(trimmed.mean()),
            float(bool(row["left_type_pair"])
                  and row["left_type_pair"] == row["right_type_pair"]),
        )
        if not all(math.isfinite(value) for value in feature):
            raise ValueError("non-finite S3 evidence feature")
        features.append(feature)
        gold.append(bool(row["same_relation"]))
    return np.asarray(features, dtype="float32"), gold


def build_features(train_set, calibration_set, train_projected, calibration_projected,
                   train_pairs, calibration_pairs, train_prior, calibration_prior):
    train_top, train_scores = neighbor_evidence(
        train_projected, train_set["dialogues"], train_projected,
        train_set["dialogues"], same_bank=True,
    )
    calibration_top, calibration_scores = neighbor_evidence(
        calibration_projected, calibration_set["dialogues"], train_projected,
        train_set["dialogues"], same_bank=False,
    )
    train_evidence = evidence_statistics(train_top, train_scores, train_set["relations"])
    calibration_evidence = evidence_statistics(
        calibration_top, calibration_scores, train_set["relations"]
    )
    train_features, train_gold = pair_feature_table(
        train_pairs, train_projected, train_set["matrix"], train_set["ids"],
        train_evidence, train_top, train_projected, train_prior,
    )
    calibration_features, calibration_gold = pair_feature_table(
        calibration_pairs, calibration_projected, calibration_set["matrix"],
        calibration_set["ids"], calibration_evidence, calibration_top,
        train_projected, calibration_prior,
    )
    return train_features, train_gold, calibration_features, calibration_gold


class FusionHead:
    @staticmethod
    def create(kind: str, dimension: int):
        import torch

        if kind == "monotone":
            class Monotone(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.raw_weight = torch.nn.Parameter(torch.zeros(dimension))
                    self.bias = torch.nn.Parameter(torch.zeros(()))

                def forward(self, value):
                    return value @ torch.nn.functional.softplus(self.raw_weight) + self.bias
            return Monotone()
        if kind == "mlp":
            return torch.nn.Sequential(
                torch.nn.Linear(dimension, 64),
                torch.nn.GELU(),
                torch.nn.Dropout(.1),
                torch.nn.Linear(64, 1),
                torch.nn.Flatten(0),
            )
        raise ValueError(kind)


def fit_head(train_features, train_gold, kind: str, columns: list[int]):
    import numpy as np
    import torch

    torch.manual_seed(SEED)
    value = torch.as_tensor(train_features[:, columns], dtype=torch.float32)
    gold = torch.as_tensor(np.asarray(train_gold, dtype="float32"))
    mean, std = value.mean(0), value.std(0).clamp_min(1e-6)
    value = (value - mean) / std
    head = FusionHead.create(kind, len(columns)).to("cuda")
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(SEED)
    last_loss = None
    for _ in range(100):
        order = torch.randperm(len(value), generator=generator)
        for start in range(0, len(order), 512):
            index = order[start:start + 512]
            logits = head(value[index].to("cuda"))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, gold[index].to("cuda")
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            last_loss = float(loss.detach())
    return head.eval(), mean, std, last_loss


def predict_head(head, features, columns, mean, std):
    import torch

    value = torch.as_tensor(features[:, columns], dtype=torch.float32)
    value = (value - mean) / std
    with torch.inference_mode():
        return head(value.to("cuda")).float().cpu().tolist()


def binary(gold: list[bool], prediction: list[bool]) -> dict:
    tp = sum(a and b for a, b in zip(gold, prediction))
    fp = sum(not a and b for a, b in zip(gold, prediction))
    fn = sum(a and not b for a, b in zip(gold, prediction))
    tn = sum(not a and not b for a, b in zip(gold, prediction))
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else 1.0,
        "recall": tp / (tp + fn) if tp + fn else 1.0,
        "specificity": tn / (tn + fp) if tn + fp else 1.0,
    }


def delete_counts(gold: list[bool], prediction: list[bool]) -> dict:
    positives = sum(gold)
    true_positive = sum(label and value for label, value in zip(gold, prediction))
    false_positive = sum(not label and value for label, value in zip(gold, prediction))
    tp, fn, fp = 2 * true_positive, 2 * (positives - true_positive), false_positive
    tn = 3 * positives - fp
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else 1.0,
        "recall": tp / (tp + fn) if tp + fn else 1.0,
        "specificity": tn / (tn + fp) if tn + fp else 1.0,
    }


def empirical_threshold(scores: list[float], gold: list[bool], precision: float = .99) -> dict:
    order = sorted(((score, index) for index, score in enumerate(scores)), reverse=True)
    prediction = [False] * len(scores)
    candidates = [(order[0][0] + 1 if order else 1.0, delete_counts(gold, prediction))]
    cursor = 0
    while cursor < len(order):
        threshold = order[cursor][0]
        while cursor < len(order) and order[cursor][0] == threshold:
            prediction[order[cursor][1]] = True
            cursor += 1
        candidates.append((threshold, delete_counts(gold, prediction)))
    feasible = [(threshold, metric) for threshold, metric in candidates
                if metric["precision"] >= precision]
    threshold, metric = max(feasible, key=lambda item: (
        item[1]["recall"], item[1]["precision"], item[0]
    ))
    return {"threshold": threshold, **metric}


def conformal_negative_threshold(scores: list[float], gold: list[bool], groups: list[str],
                                 coverage: float = .99) -> float:
    maxima = defaultdict(lambda: -math.inf)
    for score, label, group in zip(scores, gold, groups):
        if not label:
            maxima[group] = max(maxima[group], score)
    values = sorted(maxima.values())
    if not values:
        return math.inf
    index = min(len(values) - 1, max(0, math.ceil((len(values) + 1) * coverage) - 1))
    return math.nextafter(values[index], math.inf)


def macro_balanced_accuracy(pairs: list[dict], prediction: list[bool]) -> float:
    grouped = defaultdict(list)
    for row, value in zip(pairs, prediction):
        grouped[row["left_relation_audit"]].append((bool(row["same_relation"]), value))
    values = []
    for rows in grouped.values():
        metric = binary([row[0] for row in rows], [row[1] for row in rows])
        values.append((metric["recall"] + metric["specificity"]) / 2)
    return sum(values) / len(values)


def summarize_scores(pairs: list[dict], scenarios: list[dict], scores: list[float],
                     use_conformal: bool = True) -> dict:
    gold = [bool(row["same_relation"]) for row in pairs]
    empirical = empirical_threshold(scores, gold)
    conformal = conformal_negative_threshold(
        scores, gold, [row["left_dialogue_sha256"] for row in pairs]
    )
    threshold = max(empirical["threshold"], conformal) if use_conformal else empirical["threshold"]
    prediction = [score >= threshold for score in scores]
    relation = binary(gold, prediction)
    deletion = delete_counts(gold, prediction)
    pair_prediction = {row["pair_id"]: value for row, value in zip(pairs, prediction)}
    by_kind = {}
    for kind in sorted({row["kind"] for row in scenarios}):
        subset = [row for row in scenarios if row["kind"] == kind]
        correct = 0
        for row in subset:
            relation_value = pair_prediction[row["relation_pair_id"]]
            operation = ("duplicate" if row["same_entity"] and relation_value
                         and row["same_value"] else "update"
                         if row["same_entity"] and relation_value else "retain")
            correct += operation == row["operation"]
        by_kind[kind] = correct / len(subset)
    return {
        "threshold": threshold,
        "empirical_threshold": empirical["threshold"],
        "conformal_negative_threshold": conformal,
        "conformal_applied": use_conformal,
        "same_relation": relation,
        "delete": deletion,
        "relation_macro_balanced_accuracy": macro_balanced_accuracy(pairs, prediction),
        "relation_roc_auc": roc_auc(gold, scores),
        "by_kind_operation_accuracy": by_kind,
    }


def feature_columns(names: set[str]) -> list[int]:
    return [index for index, name in enumerate(FEATURE_NAMES) if name in names]


def fit(args) -> None:
    import torch

    data_root, embedding_root = Path(args.data_root), Path(args.embedding_root)
    manifest = json.loads((data_root / "manifest.json").read_text())
    audit = json.loads((data_root / "data_audit.json").read_text())
    projection = json.loads(Path(args.projection_selection).read_text())
    if (manifest.get("protocol") != DATA_PROTOCOL
            or audit.get("protocol") != AUDIT_PROTOCOL or audit.get("passed") is not True
            or projection.get("protocol") != PROJECTION_PROTOCOL
            or projection.get("dialogre_dev_loaded") is not False
            or projection.get("dialogre_test_loaded") is not False):
        raise ValueError("S3 fusion requires sealed audited inputs")
    output, weights_output, scores_output = (
        Path(args.output), Path(args.weights_output), Path(args.scores_output)
    )
    if output.exists() or weights_output.exists() or scores_output.exists():
        raise FileExistsError(output)
    train_set, _ = load_embedding_set(
        embedding_root, "train", manifest["counts"]["train"]["mentions"]
    )
    calibration_set, _ = load_embedding_set(
        embedding_root, "calibration", manifest["counts"]["calibration"]["mentions"]
    )
    train_pairs = load_jsonl(data_root / "train_relation_pairs.jsonl")
    calibration_pairs = load_jsonl(data_root / "calibration_relation_pairs.jsonl")
    calibration_scenarios = load_jsonl(data_root / "calibration_scenarios.jsonl")
    train_prior, train_prior_audits = load_prior_scores(
        args.prior_root, "train", len(train_pairs)
    )
    calibration_prior, calibration_prior_audits = load_prior_scores(
        args.prior_root, "calibration", len(calibration_pairs)
    )
    checkpoint = torch.load(args.projection_weights, map_location="cpu", weights_only=False)
    if (checkpoint.get("protocol") != PROJECTION_PROTOCOL
            or sha(args.projection_weights) != projection.get("weights_sha256")):
        raise ValueError("S3 projection weights changed")
    input_dimension = checkpoint["input_dimension"]
    selected_head = ProjectionHead.create(
        input_dimension, checkpoint["config"]["dimension"]
    )
    selected_head.load_state_dict(checkpoint["state_dict"])
    selected_head = selected_head.to("cuda")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    train_projected = project_matrix(selected_head, train_set["matrix"])
    calibration_projected = project_matrix(selected_head, calibration_set["matrix"])
    full = build_features(
        train_set, calibration_set, train_projected, calibration_projected,
        train_pairs, calibration_pairs, train_prior, calibration_prior,
    )
    train_features, train_gold, calibration_features, calibration_gold = full
    if train_gold != [bool(row["same_relation"]) for row in train_pairs]:
        raise RuntimeError("S3 train feature order mismatch")

    all_names = set(FEATURE_NAMES)
    arm_specs = {
        "monotone_full": ("monotone", all_names),
        "mlp_full": ("mlp", all_names),
        "monotone_no_s2": ("monotone", all_names - {"s2_prior_score"}),
        "mlp_no_s2": ("mlp", all_names - {"s2_prior_score"}),
        "monotone_no_support": ("monotone", DIRECT_FEATURES),
        "mlp_no_support": ("mlp", DIRECT_FEATURES),
    }
    summaries, artifacts, calibration_scores = {}, {}, {}
    for name, (kind, names) in arm_specs.items():
        columns = feature_columns(names)
        head, mean, std, loss = fit_head(train_features, train_gold, kind, columns)
        scores = predict_head(head, calibration_features, columns, mean, std)
        summaries[name] = summarize_scores(
            calibration_pairs, calibration_scenarios, scores, use_conformal=True
        )
        summaries[name]["last_training_loss"] = loss
        summaries[name]["features"] = [FEATURE_NAMES[index] for index in columns]
        calibration_scores[name] = scores
        artifacts[name] = {
            "kind": kind,
            "columns": columns,
            "mean": mean,
            "std": std,
            "state_dict": {key: value.detach().cpu() for key, value in head.state_dict().items()},
        }

    # Raw-representation support ablation.
    raw_features = build_features(
        train_set, calibration_set, train_set["matrix"], calibration_set["matrix"],
        train_pairs, calibration_pairs, train_prior, calibration_prior,
    )
    for kind in ("monotone", "mlp"):
        name = f"{kind}_no_projection"
        columns = feature_columns(all_names)
        head, mean, std, loss = fit_head(raw_features[0], raw_features[1], kind, columns)
        scores = predict_head(head, raw_features[2], columns, mean, std)
        summaries[name] = summarize_scores(
            calibration_pairs, calibration_scenarios, scores, use_conformal=True
        )
        summaries[name]["last_training_loss"] = loss
        summaries[name]["features"] = list(FEATURE_NAMES)
        calibration_scores[name] = scores
        artifacts[name] = {
            "kind": kind, "columns": columns, "mean": mean, "std": std,
            "state_dict": {key: value.detach().cpu() for key, value in head.state_dict().items()},
        }

    # PubMed pretraining ablation uses the best checkpoint from the opposite arm.
    opposite_key = str(not checkpoint["config"]["pubmed_pretraining"]).lower()
    opposite = checkpoint["pretraining_ablation"][opposite_key]
    opposite_head = ProjectionHead.create(input_dimension, opposite["config"]["dimension"])
    opposite_head.load_state_dict(opposite["state_dict"])
    opposite_head = opposite_head.to("cuda")
    opposite_features = build_features(
        train_set, calibration_set,
        project_matrix(opposite_head, train_set["matrix"]),
        project_matrix(opposite_head, calibration_set["matrix"]),
        train_pairs, calibration_pairs, train_prior, calibration_prior,
    )
    columns = feature_columns(all_names)
    head, mean, std, loss = fit_head(opposite_features[0], opposite_features[1],
                                    "monotone", columns)
    scores = predict_head(head, opposite_features[2], columns, mean, std)
    summaries["monotone_opposite_pubmed_pretraining"] = summarize_scores(
        calibration_pairs, calibration_scenarios, scores, use_conformal=True
    )
    summaries["monotone_opposite_pubmed_pretraining"]["last_training_loss"] = loss
    summaries["monotone_opposite_pubmed_pretraining"]["projection_config"] = opposite["config"]
    calibration_scores["monotone_opposite_pubmed_pretraining"] = scores

    # Non-learned and no-conformal controls use the same frozen calibration rows.
    baseline_scores = {
        "raw_cosine": calibration_features[:, FEATURE_NAMES.index("raw_direct_cosine")].tolist(),
        "projected_cosine": calibration_features[:, FEATURE_NAMES.index(
            "projected_direct_cosine")].tolist(),
        "s2_direct": calibration_features[:, FEATURE_NAMES.index("s2_prior_score")].tolist(),
        "support_posterior_dot": calibration_features[:, FEATURE_NAMES.index(
            "posterior_dot")].tolist(),
    }
    for name, scores in baseline_scores.items():
        summaries[name] = summarize_scores(
            calibration_pairs, calibration_scenarios, scores, use_conformal=True
        )
        calibration_scores[name] = scores
    eligible = ["monotone_full", "mlp_full"]
    selected_name = max(eligible, key=lambda name: (
        summaries[name]["delete"]["recall"],
        summaries[name]["relation_macro_balanced_accuracy"],
        name == "monotone_full",
    ))
    summaries[f"{selected_name}_no_conformal"] = summarize_scores(
        calibration_pairs, calibration_scenarios, calibration_scores[selected_name],
        use_conformal=False,
    )
    calibration_scores[f"{selected_name}_no_conformal"] = calibration_scores[selected_name]
    with scores_output.open("x") as sink:
        for index, row in enumerate(calibration_pairs):
            sink.write(json.dumps({
                "index": index,
                "pair_id": row["pair_id"],
                "gold_same_relation": bool(row["same_relation"]),
                "left_dialogue_sha256": row["left_dialogue_sha256"],
                "scores": {name: values[index]
                           for name, values in sorted(calibration_scores.items())},
            }, sort_keys=True) + "\n")
    torch.save({
        "protocol": PROTOCOL,
        "feature_names": FEATURE_NAMES,
        "selected_arm": selected_name,
        "selected_threshold": summaries[selected_name]["threshold"],
        "projection": checkpoint,
        "heads": artifacts,
    }, weights_output)
    seconds = time.perf_counter() - started
    result = {
        "protocol": PROTOCOL,
        "selected_arm": selected_name,
        "selected_threshold": summaries[selected_name]["threshold"],
        "selected_projection": checkpoint["config"],
        "summaries": summaries,
        "feature_names": FEATURE_NAMES,
        "selection_rule": "max deletion recall, then relation macro BA, then monotone; full heads only",
        "resource": {
            "seconds": seconds,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
        },
        "data_manifest_sha256": sha(data_root / "manifest.json"),
        "data_audit_sha256": sha(data_root / "data_audit.json"),
        "projection_selection_sha256": sha(args.projection_selection),
        "projection_weights_sha256": sha(args.projection_weights),
        "prior_audits_sha256": {
            "train": [sha(Path(args.prior_root) / f"train_shard_{shard}.audit.json")
                      for shard in range(2)],
            "calibration": [sha(Path(args.prior_root)
                                / f"calibration_shard_{shard}.audit.json")
                            for shard in range(2)],
        },
        "weights_sha256": sha(weights_output),
        "calibration_scores_sha256": sha(scores_output),
        "dialogre_dev_loaded": False,
        "dialogre_test_loaded": False,
        "pubmed_validation_loaded": False,
        "terminal_status": "fusion_selected_before_development",
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["terminal_status"],
        "selected_arm": selected_name,
        "selected": summaries[selected_name],
        "seconds": seconds,
    }, indent=2, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--embedding-root", required=True)
    parser.add_argument("--prior-root", required=True)
    parser.add_argument("--projection-selection", required=True)
    parser.add_argument("--projection-weights", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--weights-output", required=True)
    parser.add_argument("--scores-output", required=True)
    fit(parser.parse_args())


if __name__ == "__main__":
    main()
