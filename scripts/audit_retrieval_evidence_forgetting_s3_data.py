"""Independent integrity and shortcut audit for S3 construction."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


PROTOCOL = "qwen_retrieval_evidence_forgetting_s3"
SEED = 35117
CALIBRATION_FOLD = 0
BIN_WIDTH = 0.05
DIALOGRE = {
    "train.json": (2338606, "94b5799642107f9afecc5c46655d6d9612f41969893deff291b15a66298c08a0"),
    "dev.json": (751932, "3e0036a1853f33dd8fab1f008ff3c3b351f0b53e9084a6f6bce7b4c95d0f0472"),
    "test.json": (725696, "180e766323d2b5ae4763c77b2db280eeb49c909ab48c7393fbc309b7890061af"),
}
FEWREL = {
    "train_wiki.json": "8e9e58a05d559be773cbe18a8d4a16ba17e6d397663f9f05a056f894a94b92c1",
    "val_wiki.json": "76bb44c2485b6faed358a3ce040eadafcc8a8835245cd7e6afc1bef672d4a0bf",
    "val_pubmed.json": "fc0f13b534f6664fe0d8aec231d7590313bc6510d00dbc1b6746147f6a2b46e6",
    "pubmed_unsupervised.json": "fa3555941364024c2ac1b1857396515ecc65a515bc742e9da0502568fad5dd1f",
}
KINDS = {
    "paraphrased_duplicate": (1, 1, 1, "duplicate"),
    "paraphrased_update": (1, 1, 0, "update"),
    "same_relation_other_entity": (0, 1, 0, "retain"),
    "same_entity_other_relation": (1, 0, 0, "retain"),
    "unrelated": (0, 0, 0, "retain"),
}


def file_sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def source_hash(*values) -> str:
    return hashlib.sha256(json.dumps(
        values, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def stable(*values) -> str:
    return hashlib.sha256(":".join(str(value) for value in values).encode()).hexdigest()


def normalized(value: str) -> str:
    return " ".join(value.casefold().split())


def replace_arguments(dialogue: list[str], left: str, right: str) -> str:
    text = " [TURN] ".join(dialogue)
    for value, marker in sorted(((left, "[ARG1]"), (right, "[ARG2]")),
                                key=lambda item: len(item[0]), reverse=True):
        text = re.sub(r"(?<!\w)" + re.escape(value) + r"(?!\w)", marker,
                      text, flags=re.IGNORECASE)
    return "Arguments: [ARG1] ; [ARG2] . [DIALOGUE] " + " ".join(text.split())


def span(entity: list) -> tuple[int, int]:
    indices = entity[2][0]
    return min(indices), max(indices) + 1


def mask_spans(tokens: list[str], head: tuple[int, int], tail: tuple[int, int]) -> str:
    replacements = {head[0]: (head[1], "[ARG1]"), tail[0]: (tail[1], "[ARG2]")}
    output, index = [], 0
    while index < len(tokens):
        if index in replacements:
            end, marker = replacements[index]
            output.append(marker)
            index = end
        else:
            output.append(tokens[index])
            index += 1
    return " ".join(output)


def dropout(text: str, row_id: str, rate: float = .15) -> str:
    tokens, kept = text.split(), []
    for index, token in enumerate(tokens):
        if token in {"[ARG1]", "[ARG2]"}:
            kept.append(token)
        else:
            value = int(stable(SEED, "pubmed-dropout", row_id, index)[:8], 16) / 0xFFFFFFFF
            if value >= rate:
                kept.append(token)
    if not any(token not in {"[ARG1]", "[ARG2]"} for token in kept):
        kept.append(next(token for token in tokens if token not in {"[ARG1]", "[ARG2]"}))
    return " ".join(kept)


def load_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def grams(text: str) -> frozenset[str]:
    value = f"  {normalized(text)}  "
    return frozenset(value[index:index + 3] for index in range(max(1, len(value) - 2)))


def trigram(left: str, right: str) -> float:
    a, b = grams(left), grams(right)
    return len(a & b) / max(1, len(a | b))


def lexical_bin(value: float) -> int:
    return min(int(math.floor(value / BIN_WIDTH + 1e-12)), 19)


def tokens(text: str) -> list[str]:
    return re.findall(r"\[arg[12]\]|[a-z0-9_]+", text.casefold())


def bm25_pair_scores(rows: list[dict], k1: float = 1.2, b: float = .75) -> list[float]:
    corpus = [tokens(row[side]) for row in rows for side in ("left_text", "right_text")]
    average = sum(map(len, corpus)) / max(1, len(corpus))
    document_frequency = Counter()
    for document in corpus:
        document_frequency.update(set(document))
    total = len(corpus)
    idf = {term: math.log(1 + (total - count + .5) / (count + .5))
           for term, count in document_frequency.items()}

    def score(document: list[str], query: list[str]) -> float:
        frequency = Counter(document)
        norm = k1 * (1 - b + b * len(document) / max(1e-12, average))
        return sum(idf.get(term, 0.0) * frequency[term] * (k1 + 1)
                   / (frequency[term] + norm)
                   for term in set(query) if frequency[term])

    result = []
    for row in rows:
        left, right = tokens(row["left_text"]), tokens(row["right_text"])
        result.append((score(left, right) + score(right, left)) / 2)
    return result


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


def threshold_at_specificity(scores: list[float], gold: list[bool], floor: float = .95) -> dict:
    ordered = sorted(((score, index) for index, score in enumerate(scores)), reverse=True)
    prediction = [False] * len(scores)
    candidates = [(ordered[0][0] + 1 if ordered else 1.0, binary(gold, prediction))]
    cursor = 0
    while cursor < len(ordered):
        threshold = ordered[cursor][0]
        while cursor < len(ordered) and ordered[cursor][0] == threshold:
            prediction[ordered[cursor][1]] = True
            cursor += 1
        candidates.append((threshold, binary(gold, prediction)))
    feasible = [(threshold, metric) for threshold, metric in candidates
                if metric["specificity"] >= floor]
    threshold, metric = max(feasible, key=lambda item: (
        item[1]["recall"], item[1]["specificity"], item[1]["precision"], item[0]
    ))
    return {"threshold": threshold, **metric}


def flatten(value) -> bool:
    if isinstance(value, dict):
        return all(flatten(item) for item in value.values())
    return bool(value)


def raw_dialogre_mentions(root: Path) -> dict[str, dict]:
    raw = json.loads((root / "data_v2/en/data/train.json").read_text())
    expected = {}
    for dialogue_index, (dialogue, annotations) in enumerate(raw):
        dialogue_sha = canonical_sha([dialogue, annotations])
        fold = int(stable(SEED, "dialogre-fold", dialogue_sha)[:16], 16) % 5
        for annotation_index, annotation in enumerate(annotations):
            relations = sorted(set(annotation["r"]) - {"unanswerable"})
            if not relations or normalized(annotation["x"]) == normalized(annotation["y"]):
                continue
            instance_id = source_hash("dialogre-instance", dialogue_sha, annotation_index)
            masked = replace_arguments(dialogue, annotation["x"], annotation["y"])
            all_relations = [f"dialogre:{relation}" for relation in relations]
            for relation in relations:
                mention_id = source_hash("dialogre-mention", instance_id, relation)
                expected[mention_id] = {
                    "dialogue_sha256": dialogue_sha,
                    "source_instance_sha256": instance_id,
                    "fold": fold,
                    "relation": f"dialogre:{relation}",
                    "all_relations_audit": all_relations,
                    "type_pair": f"{annotation.get('x_type', '')}:{annotation.get('y_type', '')}",
                    "masked_text": masked,
                }
    return expected


def audit(data_root: str | Path, fewrel_root: str | Path,
          dialogre_root: str | Path) -> dict:
    data_root, fewrel_root, dialogre_root = map(Path, (data_root, fewrel_root, dialogre_root))
    manifest = json.loads((data_root / "manifest.json").read_text())
    checks = {
        "protocol": manifest["protocol"] == PROTOCOL,
        "configuration": manifest["seed"] == SEED
            and manifest["calibration_fold"] == CALIBRATION_FOLD
            and manifest["cap_per_relation"] == 200
            and manifest["unsupervised_dropout"] == .15,
        "model_free": manifest["model_outputs_loaded"] is False,
        "sealed": manifest["dialogre_dev_parsed"] is False
            and manifest["dialogre_test_parsed"] is False
            and manifest["pubmed_validation_parsed"] is False
            and not (data_root / "development_mentions.jsonl").exists()
            and not (data_root / "test_mentions.jsonl").exists(),
    }
    checks["dialogre_sources"] = all(
        (dialogre_root / "data_v2/en/data" / name).stat().st_size == size
        and file_sha(dialogre_root / "data_v2/en/data" / name) == digest
        for name, (size, digest) in DIALOGRE.items()
    )
    checks["fewrel_sources"] = all(
        file_sha(fewrel_root / "data" / name) == digest for name, digest in FEWREL.items()
    )

    expected_dialogre = raw_dialogre_mentions(dialogre_root)
    split_mentions = {}
    all_mention_ids = set()
    for split in ("train", "calibration"):
        path = data_root / f"{split}_mentions.jsonl"
        mentions = load_jsonl(path)
        split_mentions[split] = mentions
        valid, cap_counts = True, Counter()
        for row in mentions:
            valid &= row["source_split"] == split
            valid &= row["source_row_sha256"] not in all_mention_ids
            valid &= "[ARG1]" in row["masked_text"] and "[ARG2]" in row["masked_text"]
            cap_counts[(row["source"], row["relation"])] += 1
            if row["source"] == "dialogre":
                expected = expected_dialogre.get(row["source_row_sha256"])
                valid &= expected is not None
                if expected:
                    valid &= all(row[key] == expected[key] for key in expected)
                valid &= ((row["fold"] == CALIBRATION_FOLD) == (split == "calibration"))
            else:
                valid &= split == "train" and row["relation"].startswith("fewrel:")
            all_mention_ids.add(row["source_row_sha256"])
        counts = manifest["counts"][split]
        checks[f"{split}_mentions"] = {
            "valid": bool(valid),
            "cap": max(cap_counts.values(), default=0) <= manifest["cap_per_relation"],
            "count": len(mentions) == counts["mentions"],
            "hash": file_sha(path) == manifest["files_sha256"][path.name],
        }

    train_dialogues = {row["dialogue_sha256"] for row in split_mentions["train"]
                       if row["source"] == "dialogre"}
    calibration_dialogues = {row["dialogue_sha256"] for row in split_mentions["calibration"]}
    train_instances = {row["source_instance_sha256"] for row in split_mentions["train"]}
    calibration_instances = {row["source_instance_sha256"]
                             for row in split_mentions["calibration"]}
    train_texts = {row["masked_text"] for row in split_mentions["train"]}
    calibration_texts = {row["masked_text"] for row in split_mentions["calibration"]}
    entity_overlap = ({row["normalized_entity_pair_sha256_audit"]
                       for row in split_mentions["train"] if row["source"] == "dialogre"}
                      & {row["normalized_entity_pair_sha256_audit"]
                         for row in split_mentions["calibration"]})
    checks["split_isolation"] = {
        "dialogues": not (train_dialogues & calibration_dialogues),
        "instances": not (train_instances & calibration_instances),
        "masked_texts": not (train_texts & calibration_texts),
        "entity_overlap_reported": len(entity_overlap)
            == manifest["dialogre"]["normalized_entity_pairs_crossing_calibration_kept"],
    }

    split_pairs = {}
    for split in ("train", "calibration"):
        mention_map = {row["source_row_sha256"]: row for row in split_mentions[split]}
        pair_path = data_root / f"{split}_relation_pairs.jsonl"
        scenario_path = data_root / f"{split}_scenarios.jsonl"
        pairs, scenarios = load_jsonl(pair_path), load_jsonl(scenario_path)
        split_pairs[split] = pairs
        by_anchor, valid_pairs, pair_ids = defaultdict(list), True, set()
        for row in pairs:
            left = mention_map.get(row["left_source_row_sha256"])
            right = mention_map.get(row["right_source_row_sha256"])
            valid_pairs &= left is not None and right is not None
            if not left or not right:
                continue
            same = left["relation"] == right["relation"]
            score = trigram(left["masked_text"], right["masked_text"])
            expected_id = stable(PROTOCOL, SEED, split,
                                 "positive" if same else "negative",
                                 left["source_row_sha256"], right["source_row_sha256"])[:24]
            valid_pairs &= (
                row["pair_id"] == expected_id
                and row["label"] == ("positive" if same else "negative")
                and bool(row["same_relation"]) == same
                and left["source"] == right["source"] == row["source"]
                and left["dialogue_sha256"] != right["dialogue_sha256"]
                and left["masked_text"] != right["masked_text"]
                and abs(row["trigram_similarity"] - score) <= 1e-12
                and row["similarity_bin"] == lexical_bin(score)
            )
            if not same:
                valid_pairs &= not (set(left["all_relations_audit"])
                                    & set(right["all_relations_audit"]))
                if left["type_pair"]:
                    valid_pairs &= left["type_pair"] == right["type_pair"]
            pair_ids.add(row["pair_id"])
            by_anchor[left["source_row_sha256"]].append(row)
        matched = all(len(values) == 2
                      and {row["label"] for row in values} == {"positive", "negative"}
                      and len({row["similarity_bin"] for row in values}) == 1
                      for values in by_anchor.values())
        grouped, valid_scenarios = defaultdict(list), True
        for row in scenarios:
            expected = KINDS.get(row["kind"])
            valid_scenarios &= (expected is not None
                                and tuple(row[key] for key in
                                          ("same_entity", "same_relation", "same_value",
                                           "operation")) == expected
                                and row["relation_pair_id"] in pair_ids)
            grouped[row["group_id"]].append(row)
        complete = all(len(values) == 5 and {row["kind"] for row in values} == set(KINDS)
                       for values in grouped.values())
        counts = manifest["counts"][split]
        checks[f"{split}_pairs"] = {
            "valid": bool(valid_pairs),
            "matched_bins": matched,
            "balanced": Counter(row["label"] for row in pairs)
                == Counter({"positive": len(pairs) // 2, "negative": len(pairs) // 2}),
            "counts": len(pairs) == counts["relation_pairs"]
                and len(scenarios) == counts["scenarios"],
            "scenario_rule": bool(valid_scenarios),
            "complete_groups": complete,
            "hashes": file_sha(pair_path) == manifest["files_sha256"][pair_path.name]
                and file_sha(scenario_path) == manifest["files_sha256"][scenario_path.name],
        }

    views_path = data_root / "pubmed_unsupervised_views.jsonl"
    views = load_jsonl(views_path)
    raw_unsupervised = json.loads((fewrel_root / "data/pubmed_unsupervised.json").read_text())
    expected_views = {}
    for row in raw_unsupervised:
        full = mask_spans(row["tokens"], span(row["h"]), span(row["t"]))
        row_id = source_hash("pubmed-unsupervised", canonical_sha(row))
        expected_views[row_id] = (full, dropout(full, row_id))
    views_valid = len(views) == len(expected_views) == 2500
    for row in views:
        expected = expected_views.get(row["row_id"])
        views_valid &= expected is not None
        if expected:
            views_valid &= (row["view_full"], row["view_dropout"]) == expected
        views_valid &= row["relation_label_loaded"] is False
        views_valid &= "[ARG1]" in row["view_dropout"] and "[ARG2]" in row["view_dropout"]
    checks["pubmed_unsupervised"] = {
        "valid": bool(views_valid),
        "hash": file_sha(views_path) == manifest["files_sha256"][views_path.name],
    }

    calibration_pairs = split_pairs["calibration"]
    gold = [bool(row["same_relation"]) for row in calibration_pairs]
    trigram_result = threshold_at_specificity(
        [row["trigram_similarity"] for row in calibration_pairs], gold
    )
    bm25_result = threshold_at_specificity(bm25_pair_scores(calibration_pairs), gold)
    exact_positive_coverage = sum(
        bool(row["same_relation"]) and row["left_text"] == row["right_text"]
        for pairs in split_pairs.values() for row in pairs
    )
    shortcut = {
        "trigram_at_95_specificity": trigram_result,
        "bm25_at_95_specificity": bm25_result,
        "exact_positive_coverage": exact_positive_coverage,
    }
    checks["shortcut_gate"] = {
        "exact_positive_coverage_zero": exact_positive_coverage == 0,
        "trigram_recall_at_most_20": trigram_result["recall"] <= .20,
        "bm25_recall_at_most_20": bm25_result["recall"] <= .20,
    }
    passed = flatten(checks)
    result = {
        "protocol": "qwen_retrieval_evidence_forgetting_s3_data_audit_v1",
        "passed": passed,
        "checks": checks,
        "shortcut": shortcut,
        "rows_audited": sum(len(rows) for rows in split_mentions.values())
            + sum(len(rows) for rows in split_pairs.values()) + len(views),
        "dialogre_train_parsed": True,
        "dialogre_dev_rows_parsed": 0,
        "dialogre_test_rows_parsed": 0,
        "pubmed_validation_rows_parsed": 0,
        "manifest_sha256": file_sha(data_root / "manifest.json"),
    }
    if not passed:
        raise RuntimeError(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--fewrel-root", required=True)
    parser.add_argument("--dialogre-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(args.data_root, args.fewrel_root, args.dialogre_root)
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
