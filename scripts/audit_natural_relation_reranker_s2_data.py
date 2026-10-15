"""Independently audit S2 data integrity and the pre-model shortcut gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


PROTOCOL = "qwen_natural_relation_reranker_s2"
SEED = 25117
BIN_WIDTH = 0.05
KINDS = {
    "paraphrased_duplicate", "paraphrased_update",
    "same_relation_other_entity", "same_entity_other_relation", "unrelated",
}
FIELDS = {
    "paraphrased_duplicate": (1, 1, 1, "duplicate"),
    "paraphrased_update": (1, 1, 0, "update"),
    "same_relation_other_entity": (0, 1, 0, "retain"),
    "same_entity_other_relation": (1, 0, 0, "retain"),
    "unrelated": (0, 0, 0, "retain"),
}
FEWREL_HASHES = {
    "train_wiki.json": "8e9e58a05d559be773cbe18a8d4a16ba17e6d397663f9f05a056f894a94b92c1",
    "val_wiki.json": "76bb44c2485b6faed358a3ce040eadafcc8a8835245cd7e6afc1bef672d4a0bf",
    "val_pubmed.json": "fc0f13b534f6664fe0d8aec231d7590313bc6510d00dbc1b6746147f6a2b46e6",
}
SEMEVAL_HASHES = {
    "semeval_rel2id.json": "c3153e7d20a127c82fbc9b7542fcdc8e48714bcf9165fafd611dadaed22c7de8",
    "semeval_train.txt": "5a53be5e2dfbf865f77c287deec7b9e8a12be333ffb6157bfb5a692250a06220",
    "semeval_val.txt": "a91170e7c19354a895a58e582bf7fad43994467a0d27a8632319a70840b88e02",
    "semeval_test.txt": "1f33efe3db4dc6ff99f52416331682f3ec6759de526786d3629e6b43f8020795",
}
SEMEVAL_ROWS = {"semeval_train.txt": 6507, "semeval_val.txt": 1493,
                "semeval_test.txt": 2717}


def file_sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable(*values) -> str:
    return hashlib.sha256(":".join(str(value) for value in values).encode()).hexdigest()


def load_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def grams(text: str) -> frozenset[str]:
    value = f"  {' '.join(text.casefold().split())}  "
    return frozenset(value[index:index + 3] for index in range(max(1, len(value) - 2)))


def lexical_similarity(left: str, right: str) -> float:
    a, b = grams(left), grams(right)
    return len(a & b) / max(1, len(a | b))


def lexical_bin(value: float) -> int:
    return min(int(math.floor(value / BIN_WIDTH + 1e-12)), 19)


def binary(gold: list[bool], prediction: list[bool]) -> dict:
    tp = sum(a and b for a, b in zip(gold, prediction))
    fp = sum(not a and b for a, b in zip(gold, prediction))
    fn = sum(a and not b for a, b in zip(gold, prediction))
    tn = sum(not a and not b for a, b in zip(gold, prediction))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else 1.0,
            "recall": tp / (tp + fn) if tp + fn else 1.0,
            "specificity": tn / (tn + fp) if tn + fp else 1.0}


def select_threshold(rows: list[dict], minimum_precision: float = .99) -> dict:
    ordered = sorted(((row["trigram_similarity"], index) for index, row in enumerate(rows)),
                     reverse=True)
    gold = [bool(row["same_relation"]) for row in rows]
    prediction = [False] * len(rows)
    candidates = [(ordered[0][0] + 1.0 if ordered else 1.0, binary(gold, prediction))]
    cursor = 0
    while cursor < len(ordered):
        threshold = ordered[cursor][0]
        while cursor < len(ordered) and ordered[cursor][0] == threshold:
            prediction[ordered[cursor][1]] = True
            cursor += 1
        candidates.append((threshold, binary(gold, prediction)))
    feasible = [(threshold, value) for threshold, value in candidates
                if value["precision"] >= minimum_precision]
    threshold, value = max(feasible, key=lambda item: (
        item[1]["recall"], item[1]["precision"], item[0]))
    return {"threshold": threshold, **value}


def apply_threshold(rows: list[dict], threshold: float) -> tuple[dict, dict[str, bool]]:
    prediction = [row["trigram_similarity"] >= threshold for row in rows]
    metrics = binary([bool(row["same_relation"]) for row in rows], prediction)
    return {"threshold": threshold, **metrics}, {
        row["pair_id"]: value for row, value in zip(rows, prediction)}


def scenario_delete_metrics(rows: list[dict], relation_predictions: dict[str, bool]) -> dict:
    gold = [row["operation"] != "retain" for row in rows]
    prediction = [bool(row["same_entity"])
                  and relation_predictions[row["relation_pair_id"]] for row in rows]
    return binary(gold, prediction)


def flatten(value) -> bool:
    if isinstance(value, dict):
        return all(flatten(item) for item in value.values())
    return bool(value)


def audit(data_root: str | Path, fewrel_root: str | Path, semeval_root: str | Path) -> dict:
    data_root, fewrel_root, semeval_root = map(Path, (data_root, fewrel_root, semeval_root))
    manifest = json.loads((data_root / "manifest.json").read_text())
    checks = {
        "protocol": manifest["protocol"] == PROTOCOL,
        "construction": manifest["distinct_natural_pair_required"] is True
            and manifest["matched_similarity_bin_per_group"] is True
            and manifest["fixed_operation_rule"] is True,
        "model_free": manifest["model_outputs_loaded"] is False
            and manifest["development_model_outputs_loaded"] is False,
        "sealed": manifest["semeval_test_parsed"] is False
            and manifest["semeval_test_examples_written"] is False
            and not (data_root / "test_relation_pairs.jsonl").exists(),
    }
    checks["fewrel_hashes"] = all(file_sha(fewrel_root / "data" / name) == expected
                                   for name, expected in FEWREL_HASHES.items())
    checks["pubmed_size"] = (fewrel_root / "data/val_pubmed.json").stat().st_size == 432600
    semeval_hash_ok, semeval_rows_ok = True, True
    for name, expected in SEMEVAL_HASHES.items():
        semeval_hash_ok &= file_sha(semeval_root / name) == expected
        if name in SEMEVAL_ROWS:
            # Byte-line counting deliberately does not parse the sealed test.
            semeval_rows_ok &= sum(1 for line in (semeval_root / name).open("rb")
                                   if line.strip()) == SEMEVAL_ROWS[name]
    checks["semeval_hashes"] = semeval_hash_ok
    checks["semeval_rows"] = semeval_rows_ok

    split_data, seen_pair_ids, split_source_ids = {}, set(), {}
    for split in ("train", "calibration", "development"):
        pair_path = data_root / f"{split}_relation_pairs.jsonl"
        scenario_path = data_root / f"{split}_scenarios.jsonl"
        pairs, scenarios = load_jsonl(pair_path), load_jsonl(scenario_path)
        split_data[split] = (pairs, scenarios)
        valid_pairs, unordered, by_anchor, relation_ids = True, set(), defaultdict(list), set()
        for row in pairs:
            expected_id = stable(PROTOCOL, manifest["seed"], split, "relation", row["label"],
                                 row["left_source_row_sha256"],
                                 row["right_source_row_sha256"])[:24]
            score = lexical_similarity(row["left_text"], row["right_text"])
            same = row["left_relation_audit"] == row["right_relation_audit"]
            key = tuple(sorted((row["left_source_row_sha256"],
                                row["right_source_row_sha256"])))
            valid_pairs &= (row["split"] == split and row["pair_id"] == expected_id
                            and row["label"] == ("positive" if same else "negative")
                            and bool(row["same_relation"]) == same
                            and row["left_source_row_sha256"] != row["right_source_row_sha256"]
                            and row["left_text"] != row["right_text"]
                            and abs(score - row["trigram_similarity"]) <= 1e-12
                            and lexical_bin(score) == row["similarity_bin"]
                            and key not in unordered)
            unordered.add(key); relation_ids.add(row["pair_id"])
            by_anchor[row["left_source_row_sha256"]].append(row)
        bins_matched = all(len(values) == 2
                           and {value["label"] for value in values} == {"positive", "negative"}
                           and len({value["similarity_bin"] for value in values}) == 1
                           for values in by_anchor.values())
        grouped, valid_scenarios = defaultdict(list), True
        for row in scenarios:
            expected_id = stable(PROTOCOL, manifest["seed"], split, "scenario", row["kind"],
                                 row["left_source_row_sha256"],
                                 row["right_source_row_sha256"])[:24]
            expected = FIELDS.get(row["kind"])
            actual = (row["same_entity"], row["same_relation"],
                      row["same_value"], row["operation"])
            valid_scenarios &= (row["split"] == split and expected is not None
                                and actual == expected and row["pair_id"] == expected_id
                                and row["relation_pair_id"] in relation_ids
                                and row["left_text"] != row["right_text"]
                                and ((row["left_relation_audit"]
                                      == row["right_relation_audit"])
                                     == bool(row["same_relation"])))
            grouped[row["group_id"]].append(row)
        complete_groups = all(len(values) == 5
                              and {value["kind"] for value in values} == KINDS
                              and len({value["left_source_row_sha256"] for value in values}) == 1
                              for values in grouped.values())
        expected_counts = manifest["counts"][split]
        counts_ok = (len(by_anchor) == expected_counts["groups"]
                     and len(pairs) == expected_counts["relation_pairs"]
                     and len(scenarios) == expected_counts["scenarios"]
                     and Counter(row["label"] for row in pairs)
                     == Counter(expected_counts["by_relation_label"])
                     and Counter(row["kind"] for row in scenarios)
                     == Counter(expected_counts["by_kind"]))
        hashes_ok = (file_sha(pair_path) == manifest["files_sha256"][pair_path.name]
                     and file_sha(scenario_path) == manifest["files_sha256"][scenario_path.name])
        ids = {row["pair_id"] for row in pairs + scenarios}
        unique_ids = len(ids) == len(pairs) + len(scenarios) and not (seen_pair_ids & ids)
        seen_pair_ids.update(ids)
        split_source_ids[split] = {row["left_source_row_sha256"] for row in pairs}
        checks[f"{split}_rows"] = {
            "pairs": bool(valid_pairs), "matched_bins": bins_matched,
            "scenarios": bool(valid_scenarios), "complete_groups": complete_groups,
            "counts": counts_ok, "hashes": hashes_ok, "unique_ids": unique_ids,
        }
    checks["split_sources_disjoint"] = all(
        not (split_source_ids[a] & split_source_ids[b])
        for a, b in (("train", "calibration"), ("train", "development"),
                     ("calibration", "development")))

    calibration_pairs, _ = split_data["calibration"]
    development_pairs, development_scenarios = split_data["development"]
    threshold = select_threshold(calibration_pairs)
    development_lexical, relation_predictions = apply_threshold(
        development_pairs, threshold["threshold"])
    development_delete = scenario_delete_metrics(development_scenarios, relation_predictions)
    exact_positive_coverage = sum(
        bool(row["same_relation"]) and row["left_text"] == row["right_text"]
        for pairs, _ in split_data.values() for row in pairs)
    shortcut = {
        "calibration_threshold": threshold,
        "development_relation": development_lexical,
        "development_delete": development_delete,
        "exact_positive_coverage": exact_positive_coverage,
    }
    checks["shortcut_gate"] = {
        "exact_positive_coverage_zero": exact_positive_coverage == 0,
        "trigram_relation_recall_at_most_50": development_lexical["recall"] <= .50,
        "trigram_delete_recall_at_most_50": development_delete["recall"] <= .50,
    }
    passed = flatten(checks)
    result = {
        "protocol": "qwen_natural_relation_reranker_s2_data_audit_v1",
        "passed": passed, "checks": checks, "shortcut": shortcut,
        "rows_audited": sum(len(pairs) + len(scenarios)
                            for pairs, scenarios in split_data.values()),
        "development_source_parsed_once": True, "development_model_outputs_loaded": False,
        "semeval_test_rows_parsed": 0,
        "manifest_sha256": file_sha(data_root / "manifest.json"),
    }
    if not passed:
        raise RuntimeError(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--fewrel-root", required=True)
    parser.add_argument("--semeval-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(args.data_root, args.fewrel_root, args.semeval_root)
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
