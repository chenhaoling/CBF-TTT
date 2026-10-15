"""Build dialogue-disjoint natural relation evidence for S3.

The builder is deliberately model free.  It parses DialogRE train and the
registered PubMed-unsupervised file, but it only hashes DialogRE dev/test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from tasks.build_structured_fact_comparator_s1 import (
    canonical_sha,
    load_fewrel_train,
    primary_span,
    stable,
)
from tasks.pack_cbf_event_warmup import sha


PROTOCOL = "qwen_retrieval_evidence_forgetting_s3"
SEED = 35117
CALIBRATION_FOLD = 0
FOLDS = 5
CAP_PER_RELATION = 200
BIN_WIDTH = 0.05
NEGATIVE_POOL = 512
UNSUPERVISED_DROPOUT = 0.15

DIALOGRE_COMMIT = "b23e6cd7d811a68cf0aafa40e299143b857964e2"
DIALOGRE_FILES = {
    "train.json": (2338606, "94b5799642107f9afecc5c46655d6d9612f41969893deff291b15a66298c08a0"),
    "dev.json": (751932, "3e0036a1853f33dd8fab1f008ff3c3b351f0b53e9084a6f6bce7b4c95d0f0472"),
    "test.json": (725696, "180e766323d2b5ae4763c77b2db280eeb49c909ab48c7393fbc309b7890061af"),
}
PUBMED_UNSUPERVISED = (
    1093301,
    "fa3555941364024c2ac1b1857396515ecc65a515bc742e9da0502568fad5dd1f",
)
NO_RELATION = "unanswerable"
KINDS = (
    "paraphrased_duplicate",
    "paraphrased_update",
    "same_relation_other_entity",
    "same_entity_other_relation",
    "unrelated",
)
FIELDS = {
    "paraphrased_duplicate": (1, 1, 1, "duplicate"),
    "paraphrased_update": (1, 1, 0, "update"),
    "same_relation_other_entity": (0, 1, 0, "retain"),
    "same_entity_other_relation": (1, 0, 0, "retain"),
    "unrelated": (0, 0, 0, "retain"),
}


def normalized(value: str) -> str:
    return " ".join(value.casefold().split())


def source_hash(*values) -> str:
    return hashlib.sha256(
        json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def mask_token_spans(tokens: list[str], head: tuple[int, int], tail: tuple[int, int]) -> str:
    hs, he = head
    ts, te = tail
    if not (0 <= hs < he <= len(tokens) and 0 <= ts < te <= len(tokens)):
        raise ValueError("entity span outside token sequence")
    if not (he <= ts or te <= hs):
        raise ValueError("overlapping entity spans")
    replacements = {hs: (he, "[ARG1]"), ts: (te, "[ARG2]")}
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


def replace_dialogue_arguments(dialogue: list[str], left: str, right: str) -> str:
    """Mask case-insensitive exact argument mentions while retaining turn order."""
    if normalized(left) == normalized(right):
        raise ValueError("identical DialogRE arguments")
    text = " [TURN] ".join(dialogue)
    replacements = sorted(((left, "[ARG1]"), (right, "[ARG2]")),
                          key=lambda item: len(item[0]), reverse=True)
    for value, marker in replacements:
        boundary = r"(?<!\w)" + re.escape(value) + r"(?!\w)"
        text = re.sub(boundary, marker, text, flags=re.IGNORECASE)
    # The query header makes both argument roles explicit even when an alias is
    # not repeated verbatim in the dialogue.
    return "Arguments: [ARG1] ; [ARG2] . [DIALOGUE] " + " ".join(text.split())


def char_grams(text: str) -> frozenset[str]:
    value = f"  {normalized(text)}  "
    return frozenset(value[index:index + 3] for index in range(max(1, len(value) - 2)))


def similarity(left: frozenset[str], right: frozenset[str]) -> float:
    return len(left & right) / max(1, len(left | right))


def similarity_bin(value: float) -> int:
    return min(int(math.floor(value / BIN_WIDTH + 1e-12)), int(1 / BIN_WIDTH) - 1)


def verify_dialogre(root: str | Path) -> dict:
    root = Path(root)
    if not (root / ".git").exists():
        raise ValueError("DialogRE repository metadata missing")
    head = (root / ".git/HEAD").read_text().strip()
    # Worktrees and packed refs vary, so the runner records git rev-parse in the
    # manifest argument while file hashes remain the authoritative content lock.
    files = {}
    for name, (size, digest) in DIALOGRE_FILES.items():
        path = root / "data_v2/en/data" / name
        if path.stat().st_size != size or sha(path) != digest:
            raise ValueError(f"DialogRE source mismatch: {name}")
        files[name] = {"bytes": size, "sha256": digest}
    return {"repository_commit": DIALOGRE_COMMIT, "head_reference": head, "files": files}


def dialogue_fold(dialogue_sha256: str, seed: int = SEED) -> int:
    return int(stable(seed, "dialogre-fold", dialogue_sha256)[:16], 16) % FOLDS


def load_dialogre_train(root: str | Path, seed: int = SEED) -> tuple[list[dict], dict]:
    root = Path(root)
    metadata = verify_dialogre(root)
    raw = json.loads((root / "data_v2/en/data/train.json").read_text())
    if not isinstance(raw, list):
        raise ValueError("unexpected DialogRE train schema")
    rows, excluded = [], Counter()
    dialogues_by_fold = Counter()
    normalized_pairs_by_fold: dict[int, set[tuple[str, str]]] = defaultdict(set)
    for dialogue_index, value in enumerate(raw):
        if not (isinstance(value, list) and len(value) == 2):
            raise ValueError("unexpected DialogRE dialogue schema")
        dialogue, annotations = value
        dialogue_sha256 = canonical_sha(value)
        fold = dialogue_fold(dialogue_sha256, seed)
        dialogues_by_fold[fold] += 1
        for annotation_index, annotation in enumerate(annotations):
            relations = sorted(set(annotation["r"]))
            positive_relations = [relation for relation in relations if relation != NO_RELATION]
            if not positive_relations:
                excluded["no_relation_instance"] += 1
                continue
            left, right = str(annotation["x"]), str(annotation["y"])
            if normalized(left) == normalized(right):
                excluded["identical_arguments"] += len(positive_relations)
                continue
            try:
                masked = replace_dialogue_arguments(dialogue, left, right)
            except ValueError:
                excluded["mask_failure"] += len(positive_relations)
                continue
            instance_id = source_hash("dialogre-instance", dialogue_sha256, annotation_index)
            pair_key = tuple(sorted((normalized(left), normalized(right))))
            normalized_pairs_by_fold[fold].add(pair_key)
            all_relations = [f"dialogre:{relation}" for relation in positive_relations]
            type_pair = f"{annotation.get('x_type', '')}:{annotation.get('y_type', '')}"
            for relation in positive_relations:
                mention_id = source_hash("dialogre-mention", instance_id, relation)
                rows.append({
                    "source": "dialogre",
                    "source_split": "calibration" if fold == CALIBRATION_FOLD else "train",
                    "source_row_sha256": mention_id,
                    "source_instance_sha256": instance_id,
                    "dialogue_sha256": dialogue_sha256,
                    "dialogue_index_audit": dialogue_index,
                    "annotation_index_audit": annotation_index,
                    "fold": fold,
                    "relation": f"dialogre:{relation}",
                    "all_relations_audit": all_relations,
                    "type_pair": type_pair,
                    "normalized_entity_pair_sha256_audit": source_hash("entity-pair", pair_key),
                    "masked_text": masked,
                })
    overlap_pairs = set()
    train_pairs = set().union(*(pairs for fold, pairs in normalized_pairs_by_fold.items()
                               if fold != CALIBRATION_FOLD))
    overlap_pairs = train_pairs & normalized_pairs_by_fold[CALIBRATION_FOLD]
    metadata.update({
        "dialogues": len(raw),
        "dialogues_by_fold": dict(sorted(dialogues_by_fold.items())),
        "raw_relation_mentions": len(rows),
        "source_exclusions": dict(sorted(excluded.items())),
        "normalized_entity_pairs_crossing_calibration_raw": len(overlap_pairs),
        "dev_parsed": False,
        "test_parsed": False,
    })
    return rows, metadata


def load_fewrel_mentions(root: str | Path, cap: int = CAP_PER_RELATION,
                         seed: int = SEED) -> tuple[list[dict], dict]:
    raw, metadata = load_fewrel_train(root, cap, seed)
    rows = []
    for row in raw:
        rows.append({
            "source": "fewrel",
            "source_split": "train",
            "source_row_sha256": source_hash("fewrel-mention", row["source_row_sha256"]),
            "source_instance_sha256": source_hash("fewrel-instance", row["source_row_sha256"]),
            "dialogue_sha256": source_hash("fewrel-dialogue", row["source_row_sha256"]),
            "fold": None,
            "relation": f"fewrel:{row['relation']}",
            "all_relations_audit": [f"fewrel:{row['relation']}"],
            "type_pair": "",
            "normalized_entity_pair_sha256_audit": source_hash(
                "fewrel-entity-pair", row["head_id"], row["tail_id"]
            ),
            "masked_text": mask_token_spans(row["tokens"], row["head"], row["tail"]),
        })
    metadata["mentions"] = len(rows)
    return rows, metadata


def cap_and_filter_mentions(rows: list[dict], split: str, seed: int = SEED,
                            cap: int = CAP_PER_RELATION) -> tuple[list[dict], list[dict]]:
    by_relation: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if row["source_split"] == split:
            by_relation[(row["source"], row["relation"])].append(row)
    kept, exclusions = [], []
    for (source, relation), values in sorted(by_relation.items()):
        distinct_groups = {row["dialogue_sha256"] for row in values}
        distinct_texts = {row["masked_text"] for row in values}
        if len(distinct_groups) < 2 or len(distinct_texts) < 2:
            exclusions.extend({
                "source_row_sha256": row["source_row_sha256"],
                "relation": relation,
                "reason": "undercovered_relation",
            } for row in values)
            continue
        ordered = sorted(values, key=lambda row: stable(
            seed, "relation-cap", split, source, relation, row["source_row_sha256"]
        ))
        kept.extend(ordered[:cap])
        exclusions.extend({
            "source_row_sha256": row["source_row_sha256"],
            "relation": relation,
            "reason": "relation_cap",
        } for row in ordered[cap:])
    kept.sort(key=lambda row: row["source_row_sha256"])
    return kept, exclusions


def remove_exact_cross_split(train: list[dict], calibration: list[dict]) -> tuple[list[dict], list[dict]]:
    train_texts = {row["masked_text"] for row in train}
    kept, excluded = [], []
    for row in calibration:
        if row["masked_text"] in train_texts:
            excluded.append({
                "source_row_sha256": row["source_row_sha256"],
                "relation": row["relation"],
                "reason": "exact_masked_text_cross_split",
            })
        else:
            kept.append(row)
    return kept, excluded


def unordered_pair(left: dict, right: dict) -> tuple[str, str]:
    return tuple(sorted((left["source_row_sha256"], right["source_row_sha256"])))


def compatible_negative(anchor: dict, candidate: dict) -> bool:
    if anchor["source"] != candidate["source"]:
        return False
    if anchor["type_pair"] and candidate["type_pair"] != anchor["type_pair"]:
        return False
    return not (set(anchor["all_relations_audit"]) & set(candidate["all_relations_audit"]))


def choose_donors(rows: list[dict], by_relation: dict[str, list[dict]], anchor: dict,
                  grams: dict[str, frozenset[str]], used: set[tuple[str, str]],
                  seed: int = SEED, negative_pool: int = NEGATIVE_POOL):
    positive_bins: dict[int, list[tuple[dict, float]]] = defaultdict(list)
    for row in by_relation[anchor["relation"]]:
        if (row["dialogue_sha256"] == anchor["dialogue_sha256"]
                or row["masked_text"] == anchor["masked_text"]
                or unordered_pair(anchor, row) in used):
            continue
        score = similarity(grams[anchor["source_row_sha256"]], grams[row["source_row_sha256"]])
        positive_bins[similarity_bin(score)].append((row, score))

    start = int(stable(seed, "negative-pool", anchor["source_row_sha256"])[:16], 16) % len(rows)
    negative_bins: dict[int, list[tuple[dict, float]]] = defaultdict(list)
    seen = 0
    for offset in range(len(rows)):
        row = rows[(start + offset) % len(rows)]
        if (not compatible_negative(anchor, row)
                or row["dialogue_sha256"] == anchor["dialogue_sha256"]
                or row["masked_text"] == anchor["masked_text"]
                or unordered_pair(anchor, row) in used):
            continue
        score = similarity(grams[anchor["source_row_sha256"]], grams[row["source_row_sha256"]])
        negative_bins[similarity_bin(score)].append((row, score))
        seen += 1
        if seen >= negative_pool:
            break

    common = sorted(set(positive_bins) & set(negative_bins), reverse=True)
    if not common:
        return None
    bucket = common[0]
    positive = min(positive_bins[bucket], key=lambda item: stable(
        seed, "positive", anchor["source_row_sha256"], item[0]["source_row_sha256"]
    ))
    negative = min(negative_bins[bucket], key=lambda item: (
        abs(item[1] - positive[1]),
        stable(seed, "negative", anchor["source_row_sha256"],
               item[0]["source_row_sha256"]),
    ))
    return positive[0], positive[1], negative[0], negative[1], bucket


def pair_row(anchor: dict, donor: dict, same: bool, score: float, bucket: int,
             split: str, seed: int = SEED) -> dict:
    label = "positive" if same else "negative"
    pair_id = stable(PROTOCOL, seed, split, label, anchor["source_row_sha256"],
                     donor["source_row_sha256"])[:24]
    return {
        "pair_id": pair_id,
        "split": split,
        "source": anchor["source"],
        "label": label,
        "same_relation": int(same),
        "left_text": anchor["masked_text"],
        "right_text": donor["masked_text"],
        "left_type_pair": anchor["type_pair"],
        "right_type_pair": donor["type_pair"],
        "trigram_similarity": score,
        "similarity_bin": bucket,
        "left_source_row_sha256": anchor["source_row_sha256"],
        "right_source_row_sha256": donor["source_row_sha256"],
        "left_dialogue_sha256": anchor["dialogue_sha256"],
        "right_dialogue_sha256": donor["dialogue_sha256"],
        "left_relation_audit": anchor["relation"],
        "right_relation_audit": donor["relation"],
    }


def scenario_row(anchor: dict, donor: dict, relation_pair_id: str, kind: str,
                 split: str, seed: int = SEED) -> dict:
    entity, relation, value, operation = FIELDS[kind]
    return {
        "scenario_id": stable(PROTOCOL, seed, split, "scenario", kind,
                              anchor["source_row_sha256"],
                              donor["source_row_sha256"])[:24],
        "group_id": stable(PROTOCOL, seed, split, "group",
                           anchor["source_row_sha256"])[:24],
        "relation_pair_id": relation_pair_id,
        "split": split,
        "kind": kind,
        "same_entity": entity,
        "same_relation": relation,
        "same_value": value,
        "operation": operation,
        "left_source_row_sha256": anchor["source_row_sha256"],
        "right_source_row_sha256": donor["source_row_sha256"],
    }


def construct_pairs(rows: list[dict], split: str, seed: int = SEED) -> tuple[list[dict], list[dict], list[dict]]:
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_source[row["source"]].append(row)
    pairs, scenarios, exclusions = [], [], []
    for source, source_rows in sorted(by_source.items()):
        source_rows = sorted(source_rows, key=lambda row: row["source_row_sha256"])
        by_relation: dict[str, list[dict]] = defaultdict(list)
        for row in source_rows:
            by_relation[row["relation"]].append(row)
        grams = {row["source_row_sha256"]: char_grams(row["masked_text"])
                 for row in source_rows}
        used: set[tuple[str, str]] = set()
        for anchor in source_rows:
            chosen = choose_donors(source_rows, by_relation, anchor, grams, used, seed)
            if chosen is None:
                exclusions.append({
                    "source_row_sha256": anchor["source_row_sha256"],
                    "relation": anchor["relation"],
                    "reason": "no_matched_positive_negative_pair",
                })
                continue
            positive, positive_score, negative, negative_score, bucket = chosen
            used.add(unordered_pair(anchor, positive))
            used.add(unordered_pair(anchor, negative))
            positive_pair = pair_row(anchor, positive, True, positive_score, bucket, split, seed)
            negative_pair = pair_row(anchor, negative, False, negative_score, bucket, split, seed)
            pairs.extend((positive_pair, negative_pair))
            for kind in KINDS:
                donor, relation_pair = ((positive, positive_pair)
                                        if FIELDS[kind][1] else (negative, negative_pair))
                scenarios.append(scenario_row(anchor, donor, relation_pair["pair_id"],
                                              kind, split, seed))
    pairs.sort(key=lambda row: row["pair_id"])
    scenarios.sort(key=lambda row: row["scenario_id"])
    return pairs, scenarios, exclusions


def deterministic_dropout(text: str, row_id: str, seed: int = SEED,
                          rate: float = UNSUPERVISED_DROPOUT) -> str:
    tokens = text.split()
    kept = []
    for index, token in enumerate(tokens):
        if token in {"[ARG1]", "[ARG2]"}:
            kept.append(token)
            continue
        value = int(stable(seed, "pubmed-dropout", row_id, index)[:8], 16) / 0xFFFFFFFF
        if value >= rate:
            kept.append(token)
    non_markers = [token for token in kept if token not in {"[ARG1]", "[ARG2]"}]
    if not non_markers:
        for token in tokens:
            if token not in {"[ARG1]", "[ARG2]"}:
                kept.append(token)
                break
    return " ".join(kept)


def load_pubmed_unsupervised(root: str | Path, seed: int = SEED) -> tuple[list[dict], dict]:
    path = Path(root) / "data/pubmed_unsupervised.json"
    expected_size, expected_sha = PUBMED_UNSUPERVISED
    if path.stat().st_size != expected_size or sha(path) != expected_sha:
        raise ValueError("PubMed-unsupervised source mismatch")
    raw = json.loads(path.read_text())
    if not isinstance(raw, list) or len(raw) != 2500:
        raise ValueError("unexpected PubMed-unsupervised schema")
    rows = []
    for index, row in enumerate(raw):
        head, tail = primary_span(row["h"]), primary_span(row["t"])
        masked = mask_token_spans(row["tokens"], head, tail)
        row_id = source_hash("pubmed-unsupervised", canonical_sha(row))
        view = deterministic_dropout(masked, row_id, seed)
        if "[ARG1]" not in view or "[ARG2]" not in view:
            raise RuntimeError("entity-preserving dropout removed a marker")
        rows.append({
            "row_id": row_id,
            "source_index_audit": index,
            "view_full": masked,
            "view_dropout": view,
            "dropout_rate": UNSUPERVISED_DROPOUT,
            "relation_label_loaded": False,
        })
    return rows, {"bytes": expected_size, "sha256": expected_sha, "rows": len(rows)}


def write_jsonl(path: Path, rows: list[dict]) -> str:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                            for row in rows))
    return sha(path)


def split_summary(mentions: list[dict], pairs: list[dict], scenarios: list[dict]) -> dict:
    return {
        "mentions": len(mentions),
        "relations": len({row["relation"] for row in mentions}),
        "sources": dict(sorted(Counter(row["source"] for row in mentions).items())),
        "relation_pairs": len(pairs),
        "relation_pair_labels": dict(sorted(Counter(row["label"] for row in pairs).items())),
        "scenarios": len(scenarios),
        "scenario_kinds": dict(sorted(Counter(row["kind"] for row in scenarios).items())),
        "similarity_bins": dict(sorted(Counter(str(row["similarity_bin"])
                                                for row in pairs).items())),
    }


def build(fewrel_root: str | Path, dialogre_root: str | Path, output: str | Path,
          seed: int = SEED) -> dict:
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    fewrel_rows, fewrel_meta = load_fewrel_mentions(fewrel_root, CAP_PER_RELATION, seed)
    dialogre_rows, dialogre_meta = load_dialogre_train(dialogre_root, seed)
    pubmed_views, pubmed_meta = load_pubmed_unsupervised(fewrel_root, seed)

    train, train_source_exclusions = cap_and_filter_mentions(
        fewrel_rows + dialogre_rows, "train", seed
    )
    calibration, calibration_source_exclusions = cap_and_filter_mentions(
        dialogre_rows, "calibration", seed
    )
    calibration, cross_split_exclusions = remove_exact_cross_split(train, calibration)
    calibration_source_exclusions.extend(cross_split_exclusions)
    train_entity_pairs = {
        row["normalized_entity_pair_sha256_audit"] for row in train
        if row["source"] == "dialogre"
    }
    calibration_entity_pairs = {
        row["normalized_entity_pair_sha256_audit"] for row in calibration
    }
    dialogre_meta["normalized_entity_pairs_crossing_calibration_kept"] = len(
        train_entity_pairs & calibration_entity_pairs
    )
    train_pairs, train_scenarios, train_pair_exclusions = construct_pairs(train, "train", seed)
    calibration_pairs, calibration_scenarios, calibration_pair_exclusions = construct_pairs(
        calibration, "calibration", seed
    )
    if not train_pairs or not calibration_pairs:
        raise RuntimeError("S3 constructed an empty relation-pair split")

    output.mkdir(parents=True)
    products = {
        "train_mentions.jsonl": train,
        "train_relation_pairs.jsonl": train_pairs,
        "train_scenarios.jsonl": train_scenarios,
        "calibration_mentions.jsonl": calibration,
        "calibration_relation_pairs.jsonl": calibration_pairs,
        "calibration_scenarios.jsonl": calibration_scenarios,
        "pubmed_unsupervised_views.jsonl": pubmed_views,
    }
    files = {name: write_jsonl(output / name, rows) for name, rows in products.items()}
    manifest = {
        "protocol": PROTOCOL,
        "seed": seed,
        "calibration_fold": CALIBRATION_FOLD,
        "folds": FOLDS,
        "cap_per_relation": CAP_PER_RELATION,
        "bin_width": BIN_WIDTH,
        "negative_pool": NEGATIVE_POOL,
        "unsupervised_dropout": UNSUPERVISED_DROPOUT,
        "fewrel": fewrel_meta,
        "dialogre": dialogre_meta,
        "pubmed_unsupervised": pubmed_meta,
        "counts": {
            "train": split_summary(train, train_pairs, train_scenarios),
            "calibration": split_summary(
                calibration, calibration_pairs, calibration_scenarios
            ),
        },
        "source_exclusions": {
            "train": train_source_exclusions,
            "calibration": calibration_source_exclusions,
        },
        "pair_exclusions": {
            "train": train_pair_exclusions,
            "calibration": calibration_pair_exclusions,
        },
        "files_sha256": files,
        "matched_similarity_bins": True,
        "dialogue_disjoint": True,
        "source_instance_disjoint": True,
        "exact_masked_text_disjoint": True,
        "entity_pair_overlap_reported": True,
        "model_outputs_loaded": False,
        "pubmed_validation_parsed": False,
        "dialogre_dev_parsed": False,
        "dialogre_test_parsed": False,
        "dialogre_dev_examples_written": False,
        "dialogre_test_examples_written": False,
        "terminal_status": "built_before_model_execution",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fewrel-root", required=True)
    parser.add_argument("--dialogre-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    print(json.dumps(build(args.fewrel_root, args.dialogre_root, args.output, args.seed),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
