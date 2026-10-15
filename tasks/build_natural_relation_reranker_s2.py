"""Build shortcut-controlled natural relation pairs for S2."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from tasks.build_structured_fact_comparator_s1 import (
    SEMEVAL_HASHES, canonical_sha, load_fewrel_train, load_semeval,
    operation, primary_span, stable, verify_semeval,
)
from tasks.pack_cbf_event_warmup import sha


PROTOCOL = "qwen_natural_relation_reranker_s2"
SEED = 25117
BIN_WIDTH = 0.05
NEGATIVE_POOL = 256
PUBMED_HASH = "fc0f13b534f6664fe0d8aec231d7590313bc6510d00dbc1b6746147f6a2b46e6"
PUBMED_BYTES = 432600
PUBMED_RELATIONS = 10
PUBMED_ROWS = 1000
KINDS = (
    "paraphrased_duplicate", "paraphrased_update",
    "same_relation_other_entity", "same_entity_other_relation", "unrelated",
)
FIELDS = {
    "paraphrased_duplicate": (1, 1, 1),
    "paraphrased_update": (1, 1, 0),
    "same_relation_other_entity": (0, 1, 0),
    "same_entity_other_relation": (1, 0, 0),
    "unrelated": (0, 0, 0),
}


def mask_spans(tokens: list[str], head: tuple[int, int], tail: tuple[int, int]) -> str:
    """Replace entity strings, preserving only their roles and natural context."""
    hs, he = head
    ts, te = tail
    if not (0 <= hs < he <= len(tokens) and 0 <= ts < te <= len(tokens)):
        raise ValueError("entity span outside sentence")
    if not (he <= ts or te <= hs):
        raise ValueError("overlapping entity spans")
    replacements = {hs: (he, "[SUBJECT]"), ts: (te, "[VALUE]")}
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


def add_masked(row: dict) -> dict:
    row = dict(row)
    row["masked_text"] = mask_spans(row["tokens"], row["head"], row["tail"])
    return row


def char_grams(text: str) -> frozenset[str]:
    value = f"  {' '.join(text.casefold().split())}  "
    return frozenset(value[index:index + 3] for index in range(max(1, len(value) - 2)))


def similarity(left: frozenset[str], right: frozenset[str]) -> float:
    return len(left & right) / max(1, len(left | right))


def similarity_bin(value: float) -> int:
    return min(int(math.floor(value / BIN_WIDTH + 1e-12)), int(1 / BIN_WIDTH) - 1)


def load_pubmed(root: str | Path) -> tuple[list[dict], dict]:
    path = Path(root) / "data/val_pubmed.json"
    if sha(path) != PUBMED_HASH or path.stat().st_size != PUBMED_BYTES:
        raise ValueError("FewRel PubMed source mismatch")
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict) or len(raw) != PUBMED_RELATIONS:
        raise ValueError("unexpected PubMed relation inventory")
    rows = []
    for relation in sorted(raw):
        if len(raw[relation]) != 100:
            raise ValueError(f"unexpected PubMed rows for {relation}")
        for index, value in enumerate(raw[relation]):
            head, tail = primary_span(value["h"]), primary_span(value["t"])
            rows.append(add_masked({
                "source": "fewrel_pubmed", "source_split": "development",
                "relation": relation, "source_index": index,
                "source_row_sha256": canonical_sha(value), "tokens": value["tokens"],
                "head": head, "tail": tail,
            }))
    if len(rows) != PUBMED_ROWS:
        raise ValueError("unexpected PubMed total rows")
    return rows, {"sha256": PUBMED_HASH, "bytes": PUBMED_BYTES,
                  "relations": len(raw), "rows": len(rows)}


def filter_natural_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Keep relations with two distinct natural masked contexts."""
    by_relation = defaultdict(list)
    for row in rows:
        by_relation[row["relation"]].append(row)
    kept, excluded = [], []
    for relation, values in sorted(by_relation.items()):
        if len({row["masked_text"] for row in values}) >= 2:
            kept.extend(values)
        else:
            excluded.extend({"source_row_sha256": row["source_row_sha256"],
                             "relation": relation,
                             "reason": "fewer_than_two_distinct_masked_contexts"}
                            for row in values)
    return kept, excluded


def unordered_key(left: dict, right: dict) -> tuple[str, str]:
    return tuple(sorted((left["source_row_sha256"], right["source_row_sha256"])))


def choose_donors(rows: list[dict], by_relation: dict[str, list[dict]], anchor: dict,
                  grams: dict[str, frozenset[str]], used: set[tuple[str, str]],
                  seed: int, negative_pool: int = NEGATIVE_POOL):
    """Choose one positive and one negative in the same lexical-similarity bin."""
    positive_bins = defaultdict(list)
    for row in by_relation[anchor["relation"]]:
        if (row["source_row_sha256"] == anchor["source_row_sha256"]
                or row["masked_text"] == anchor["masked_text"]
                or unordered_key(anchor, row) in used):
            continue
        score = similarity(grams[anchor["source_row_sha256"]],
                           grams[row["source_row_sha256"]])
        positive_bins[similarity_bin(score)].append((row, score))

    start = int(stable(seed, "negative-pool", anchor["source_row_sha256"])[:16], 16) % len(rows)
    negative_bins = defaultdict(list)
    seen = 0
    for offset in range(len(rows)):
        row = rows[(start + offset) % len(rows)]
        if (row["relation"] == anchor["relation"]
                or row["masked_text"] == anchor["masked_text"]
                or unordered_key(anchor, row) in used):
            continue
        score = similarity(grams[anchor["source_row_sha256"]],
                           grams[row["source_row_sha256"]])
        negative_bins[similarity_bin(score)].append((row, score))
        seen += 1
        if seen == negative_pool:
            break

    common = sorted(set(positive_bins) & set(negative_bins), reverse=True)
    if not common:
        return None
    bucket = common[0]
    positive = min(positive_bins[bucket], key=lambda item: stable(
        seed, "positive", anchor["source_row_sha256"], item[0]["source_row_sha256"]))
    negative = min(negative_bins[bucket], key=lambda item: (
        abs(item[1] - positive[1]), stable(
            seed, "negative", anchor["source_row_sha256"], item[0]["source_row_sha256"])))
    return positive[0], positive[1], negative[0], negative[1], bucket


def relation_pair(anchor: dict, donor: dict, same_relation: bool, score: float,
                  bucket: int, split: str, seed: int) -> dict:
    label = "positive" if same_relation else "negative"
    pair_id = stable(PROTOCOL, seed, split, "relation", label,
                     anchor["source_row_sha256"], donor["source_row_sha256"])[:24]
    return {
        "pair_id": pair_id, "split": split, "label": label,
        "same_relation": int(same_relation),
        "left_text": anchor["masked_text"], "right_text": donor["masked_text"],
        "trigram_similarity": score, "similarity_bin": bucket,
        "left_source_row_sha256": anchor["source_row_sha256"],
        "right_source_row_sha256": donor["source_row_sha256"],
        "left_relation_audit": anchor["relation"],
        "right_relation_audit": donor["relation"],
    }


def scenario(anchor: dict, donor: dict, relation_pair_id: str, kind: str,
             split: str, seed: int) -> dict:
    entity, relation, value = FIELDS[kind]
    return {
        "pair_id": stable(PROTOCOL, seed, split, "scenario", kind,
                          anchor["source_row_sha256"], donor["source_row_sha256"])[:24],
        "group_id": stable(PROTOCOL, seed, split, "group",
                           anchor["source_row_sha256"])[:24],
        "relation_pair_id": relation_pair_id, "split": split, "kind": kind,
        "left_text": anchor["masked_text"], "right_text": donor["masked_text"],
        "same_entity": entity, "same_relation": relation, "same_value": value,
        "operation": operation(bool(entity), bool(relation), bool(value)),
        "left_source_row_sha256": anchor["source_row_sha256"],
        "right_source_row_sha256": donor["source_row_sha256"],
        "left_relation_audit": anchor["relation"],
        "right_relation_audit": donor["relation"],
    }


def construct(rows: list[dict], split: str, seed: int = SEED,
              negative_pool: int = NEGATIVE_POOL) -> tuple[list[dict], list[dict], list[dict]]:
    rows = sorted(rows, key=lambda row: row["source_row_sha256"])
    by_relation = defaultdict(list)
    for row in rows:
        by_relation[row["relation"]].append(row)
    grams = {row["source_row_sha256"]: char_grams(row["masked_text"]) for row in rows}
    used, relation_pairs, scenarios, excluded = set(), [], [], []
    for anchor in rows:
        selected = choose_donors(rows, by_relation, anchor, grams, used, seed, negative_pool)
        if selected is None:
            excluded.append({"source_row_sha256": anchor["source_row_sha256"],
                             "relation": anchor["relation"],
                             "reason": "no_unused_positive_negative_common_similarity_bin"})
            continue
        positive, positive_score, negative, negative_score, bucket = selected
        used.add(unordered_key(anchor, positive)); used.add(unordered_key(anchor, negative))
        pos = relation_pair(anchor, positive, True, positive_score, bucket, split, seed)
        neg = relation_pair(anchor, negative, False, negative_score, bucket, split, seed)
        relation_pairs.extend((pos, neg))
        for kind in KINDS:
            donor, pair = ((positive, pos) if FIELDS[kind][1] else (negative, neg))
            scenarios.append(scenario(anchor, donor, pair["pair_id"], kind, split, seed))
    relation_pairs.sort(key=lambda row: row["pair_id"])
    scenarios.sort(key=lambda row: row["pair_id"])
    if len({row["pair_id"] for row in relation_pairs}) != len(relation_pairs):
        raise RuntimeError("relation-pair ID collision")
    if len({row["pair_id"] for row in scenarios}) != len(scenarios):
        raise RuntimeError("scenario ID collision")
    return relation_pairs, scenarios, excluded


def summary(relation_pairs: list[dict], scenarios: list[dict]) -> dict:
    return {
        "groups": len(scenarios) // len(KINDS),
        "relation_pairs": len(relation_pairs), "scenarios": len(scenarios),
        "by_relation_label": dict(sorted(Counter(row["label"] for row in relation_pairs).items())),
        "by_kind": dict(sorted(Counter(row["kind"] for row in scenarios).items())),
        "similarity_bins": dict(sorted(Counter(str(row["similarity_bin"])
                                                for row in relation_pairs).items())),
        "left_relations": len({row["left_relation_audit"] for row in relation_pairs}),
    }


def write_jsonl(path: Path, rows: list[dict]) -> str:
    path.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
                            for row in rows))
    return sha(path)


def build(fewrel_root: str | Path, semeval_root: str | Path, output: str | Path,
          fewrel_cap: int = 200, seed: int = SEED) -> dict:
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    semeval_counts = verify_semeval(semeval_root)
    train, fewrel_meta = load_fewrel_train(fewrel_root, fewrel_cap, seed)
    calibration = load_semeval(semeval_root, "semeval_train.txt", "calibration")
    development, pubmed_meta = load_pubmed(fewrel_root)
    raw = {"train": [add_masked(row) for row in train],
           "calibration": [add_masked(row) for row in calibration],
           "development": development}
    source_exclusions, pair_exclusions, products = {}, {}, {}
    for split, rows in raw.items():
        rows, source_exclusions[split] = filter_natural_rows(rows)
        relation_pairs, scenarios, pair_exclusions[split] = construct(rows, split, seed)
        products[split] = (relation_pairs, scenarios)
        if not relation_pairs:
            raise RuntimeError(f"S2 produced no {split} relation pairs")

    output.mkdir(parents=True)
    files = {}
    for split, (relation_pairs, scenarios) in products.items():
        files[f"{split}_relation_pairs.jsonl"] = write_jsonl(
            output / f"{split}_relation_pairs.jsonl", relation_pairs)
        files[f"{split}_scenarios.jsonl"] = write_jsonl(
            output / f"{split}_scenarios.jsonl", scenarios)
    manifest = {
        "protocol": PROTOCOL, "seed": seed, "bin_width": BIN_WIDTH,
        "negative_pool": NEGATIVE_POOL, "fewrel": fewrel_meta,
        "semeval_source_sha256": SEMEVAL_HASHES,
        "semeval_source_rows": semeval_counts, "pubmed": pubmed_meta,
        "counts": {split: summary(*value) for split, value in products.items()},
        "source_exclusions": source_exclusions, "pair_exclusions": pair_exclusions,
        "files_sha256": files, "kinds": KINDS,
        "distinct_natural_pair_required": True,
        "matched_similarity_bin_per_group": True,
        "fixed_operation_rule": True,
        "semeval_test_parsed": False, "semeval_test_examples_written": False,
        "model_outputs_loaded": False, "development_model_outputs_loaded": False,
        "terminal_status": "built_before_model_execution",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fewrel-root", required=True)
    parser.add_argument("--semeval-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fewrel-cap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    print(json.dumps(build(args.fewrel_root, args.semeval_root, args.output,
                           args.fewrel_cap, args.seed), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
