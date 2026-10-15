"""Independently reconstruct S3 fusion calibration metrics and selection."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from tasks.pack_cbf_event_warmup import sha
from tasks.train_retrieval_evidence_forgetting_s3_projection import roc_auc


PROTOCOL = "qwen_retrieval_evidence_forgetting_s3_fusion_v1"


def load_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def binary(gold, prediction):
    tp = sum(a and b for a, b in zip(gold, prediction))
    fp = sum(not a and b for a, b in zip(gold, prediction))
    fn = sum(a and not b for a, b in zip(gold, prediction))
    tn = sum(not a and not b for a, b in zip(gold, prediction))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else 1.0,
            "recall": tp / (tp + fn) if tp + fn else 1.0,
            "specificity": tn / (tn + fp) if tn + fp else 1.0}


def deletion(gold, prediction):
    positives = sum(gold)
    selected_positive = sum(a and b for a, b in zip(gold, prediction))
    selected_negative = sum(not a and b for a, b in zip(gold, prediction))
    tp, fn, fp = 2 * selected_positive, 2 * (positives - selected_positive), selected_negative
    tn = 3 * positives - fp
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else 1.0,
            "recall": tp / (tp + fn) if tp + fn else 1.0,
            "specificity": tn / (tn + fp) if tn + fp else 1.0}


def empirical(scores, gold):
    order = sorted(((value, index) for index, value in enumerate(scores)), reverse=True)
    prediction = [False] * len(scores)
    candidates = [(order[0][0] + 1, deletion(gold, prediction))]
    cursor = 0
    while cursor < len(order):
        threshold = order[cursor][0]
        while cursor < len(order) and order[cursor][0] == threshold:
            prediction[order[cursor][1]] = True; cursor += 1
        metric = deletion(gold, prediction)
        if metric["precision"] >= .99:
            candidates.append((threshold, metric))
    return max(candidates, key=lambda item: (
        item[1]["recall"], item[1]["precision"], item[0]
    ))[0]


def conformal(scores, gold, groups):
    maxima = defaultdict(lambda: -math.inf)
    for score, label, group in zip(scores, gold, groups):
        if not label:
            maxima[group] = max(maxima[group], score)
    values = sorted(maxima.values())
    index = min(len(values) - 1, max(0, math.ceil((len(values) + 1) * .99) - 1))
    return math.nextafter(values[index], math.inf)


def reconstruct(pairs, scenarios, scores, use_conformal):
    gold = [bool(row["same_relation"]) for row in pairs]
    empirical_value = empirical(scores, gold)
    conformal_value = conformal(scores, gold, [row["left_dialogue_sha256"] for row in pairs])
    threshold = max(empirical_value, conformal_value) if use_conformal else empirical_value
    prediction = [score >= threshold for score in scores]
    relation = binary(gold, prediction)
    pair_prediction = {row["pair_id"]: value for row, value in zip(pairs, prediction)}
    by_kind = {}
    for kind in sorted({row["kind"] for row in scenarios}):
        rows = [row for row in scenarios if row["kind"] == kind]
        correct = 0
        for row in rows:
            relation_value = pair_prediction[row["relation_pair_id"]]
            operation = ("duplicate" if row["same_entity"] and relation_value
                         and row["same_value"] else "update"
                         if row["same_entity"] and relation_value else "retain")
            correct += operation == row["operation"]
        by_kind[kind] = correct / len(rows)
    relation_groups = defaultdict(list)
    for row, value in zip(pairs, prediction):
        relation_groups[row["left_relation_audit"]].append((bool(row["same_relation"]), value))
    macro = sum((binary([x[0] for x in rows], [x[1] for x in rows])["recall"]
                 + binary([x[0] for x in rows], [x[1] for x in rows])["specificity"]) / 2
                for rows in relation_groups.values()) / len(relation_groups)
    return {
        "threshold": threshold,
        "empirical_threshold": empirical_value,
        "conformal_negative_threshold": conformal_value,
        "conformal_applied": use_conformal,
        "same_relation": relation,
        "delete": deletion(gold, prediction),
        "relation_macro_balanced_accuracy": macro,
        "relation_roc_auc": roc_auc(gold, scores),
        "by_kind_operation_accuracy": by_kind,
    }


def close(left, right, tolerance=1e-8):
    if isinstance(left, dict):
        return set(left) <= set(right) and all(close(value, right[key], tolerance)
                                                for key, value in left.items())
    if isinstance(left, float):
        return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)
    return left == right


def audit(data_root: str | Path, fusion_root: str | Path) -> dict:
    import torch

    data_root, fusion_root = Path(data_root), Path(fusion_root)
    selection_path = fusion_root / "selection.json"
    weights_path = fusion_root / "fusion.pt"
    scores_path = fusion_root / "calibration_scores.jsonl"
    selection = json.loads(selection_path.read_text())
    weights = torch.load(weights_path, map_location="cpu", weights_only=False)
    pairs = load_jsonl(data_root / "calibration_relation_pairs.jsonl")
    scenarios = load_jsonl(data_root / "calibration_scenarios.jsonl")
    score_rows = load_jsonl(scores_path)
    row_valid = (len(score_rows) == len(pairs)
                 and [row["index"] for row in score_rows] == list(range(len(pairs)))
                 and [row["pair_id"] for row in score_rows]
                 == [row["pair_id"] for row in pairs])
    arm_names = set(score_rows[0]["scores"])
    reconstructed, summaries_ok = {}, True
    for name in arm_names:
        scores = [row["scores"][name] for row in score_rows]
        use_conformal = not name.endswith("_no_conformal")
        reconstructed[name] = reconstruct(pairs, scenarios, scores, use_conformal)
        summaries_ok &= close(reconstructed[name], selection["summaries"][name])
    eligible = ["monotone_full", "mlp_full"]
    selected = max(eligible, key=lambda name: (
        reconstructed[name]["delete"]["recall"],
        reconstructed[name]["relation_macro_balanced_accuracy"],
        name == "monotone_full",
    ))
    checks = {
        "protocol": selection.get("protocol") == PROTOCOL
            and weights.get("protocol") == PROTOCOL,
        "raw_scores": row_valid and all(
            math.isfinite(value) for row in score_rows for value in row["scores"].values()
        ),
        "score_hash": sha(scores_path) == selection.get("calibration_scores_sha256"),
        "summaries": summaries_ok,
        "selection": selected == selection["selected_arm"] == weights["selected_arm"]
            and math.isclose(selection["selected_threshold"],
                             reconstructed[selected]["threshold"], abs_tol=1e-8),
        "weights": sha(weights_path) == selection["weights_sha256"]
            and tuple(weights["feature_names"]) == tuple(selection["feature_names"]),
        "resource": math.isfinite(selection["resource"]["seconds"])
            and selection["resource"]["peak_reserved_gib"] < 28,
        "provenance": selection["data_manifest_sha256"] == sha(data_root / "manifest.json")
            and selection["data_audit_sha256"] == sha(data_root / "data_audit.json"),
        "sealed": selection.get("dialogre_dev_loaded") is False
            and selection.get("dialogre_test_loaded") is False
            and selection.get("pubmed_validation_loaded") is False,
    }
    passed = all(checks.values())
    result = {
        "protocol": "qwen_retrieval_evidence_forgetting_s3_fusion_audit_v1",
        "passed": passed,
        "checks": checks,
        "selected_arm": selected,
        "selected": reconstructed[selected],
        "selection_sha256": sha(selection_path),
        "weights_sha256": sha(weights_path),
        "scores_sha256": sha(scores_path),
        "rows_reconstructed": len(pairs),
        "dialogre_dev_rows_parsed": 0,
        "dialogre_test_rows_parsed": 0,
    }
    if not passed:
        raise RuntimeError(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--fusion-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(args.data_root, args.fusion_root)
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
