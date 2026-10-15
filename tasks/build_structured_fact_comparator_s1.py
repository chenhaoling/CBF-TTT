"""Build S1 structured fact-comparison data without opening any test row."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from tasks.build_fewrel_selective_forgetting import (
    SOURCE_COMMIT, SOURCE_HASHES, relation_partitions, span_indices,
)
from tasks.pack_cbf_event_warmup import sha


PROTOCOL = "qwen_structured_fact_comparator_s1"
SEED = 24117
KINDS = (
    "duplicate", "update", "same_relation_other_entity",
    "same_entity_other_relation", "unrelated",
)
SEMEVAL_HASHES = {
    "semeval_rel2id.json": "c3153e7d20a127c82fbc9b7542fcdc8e48714bcf9165fafd611dadaed22c7de8",
    "semeval_train.txt": "5a53be5e2dfbf865f77c287deec7b9e8a12be333ffb6157bfb5a692250a06220",
    "semeval_val.txt": "a91170e7c19354a895a58e582bf7fad43994467a0d27a8632319a70840b88e02",
    "semeval_test.txt": "1f33efe3db4dc6ff99f52416331682f3ec6759de526786d3629e6b43f8020795",
}
SEMEVAL_ROWS = {"semeval_train.txt": 6507, "semeval_val.txt": 1493,
                "semeval_test.txt": 2717}


def stable(*values) -> str:
    return hashlib.sha256(":".join(str(value) for value in values).encode()).hexdigest()


def normalize(value: str) -> str:
    return " ".join(value.casefold().split())


def canonical_sha(value) -> str:
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def render(tokens: list[str], head: tuple[int, int], tail: tuple[int, int],
           head_value: str | None = None, tail_value: str | None = None) -> str:
    """Render non-overlapping half-open entity spans with stable explicit tags."""
    hs, he = head; ts, te = tail
    if not (0 <= hs < he <= len(tokens) and 0 <= ts < te <= len(tokens)):
        raise ValueError("entity span outside sentence")
    if not (he <= ts or te <= hs):
        raise ValueError("overlapping entity spans")
    replacements = {
        hs: (he, "[SUBJECT]", head_value or " ".join(tokens[hs:he]), "[/SUBJECT]"),
        ts: (te, "[VALUE]", tail_value or " ".join(tokens[ts:te]), "[/VALUE]"),
    }
    output, index = [], 0
    while index < len(tokens):
        if index in replacements:
            end, left, value, right = replacements[index]
            output.extend((left, value, right)); index = end
        else:
            output.append(tokens[index]); index += 1
    return " ".join(output)


def primary_span(entity: list) -> tuple[int, int]:
    """Convert FewRel's inclusive token-index mention to a half-open span."""
    mentions = span_indices(entity)
    indices = mentions[0]
    if tuple(range(min(indices), max(indices) + 1)) != tuple(indices):
        raise ValueError("non-contiguous primary FewRel entity span")
    return min(indices), max(indices) + 1


def _fewrel_row(relation: str, index: int, row: dict) -> dict:
    head, tail = primary_span(row["h"]), primary_span(row["t"])
    tokens = row["tokens"]
    head_text, tail_text = " ".join(tokens[head[0]:head[1]]), " ".join(tokens[tail[0]:tail[1]])
    source_hash = canonical_sha(row)
    return {
        "source": "fewrel", "source_split": "train", "relation": relation,
        "source_index": index, "source_row_sha256": source_hash,
        "tokens": tokens, "head": head, "tail": tail,
        "head_text": head_text, "tail_text": tail_text,
        "head_id": str(row["h"][1]), "tail_id": str(row["t"][1]),
        "text": render(tokens, head, tail),
    }


def load_fewrel_train(root: str | Path, cap: int, seed: int = SEED) -> tuple[list[dict], dict]:
    root = Path(root)
    if sha(root / "data/train_wiki.json") != SOURCE_HASHES["train_wiki.json"]:
        raise ValueError("FewRel train hash mismatch")
    if sha(root / "data/val_wiki.json") != SOURCE_HASHES["val_wiki.json"]:
        raise ValueError("FewRel validation hash mismatch")
    train = json.loads((root / "data/train_wiki.json").read_text())
    external = json.loads((root / "data/val_wiki.json").read_text())
    partitions = relation_partitions(train, external, seed=2311)
    relations = set(partitions["train"])
    rows = []
    for relation in sorted(relations):
        candidates = {_row["source_row_sha256"]: _row
                      for index, row in enumerate(train[relation])
                      for _row in [_fewrel_row(relation, index, row)]}
        candidates = list(candidates.values())
        candidates.sort(key=lambda row: stable(seed, "fewrel-cap", row["source_row_sha256"]))
        rows.extend(candidates[:cap])
    return rows, {"repository_commit": SOURCE_COMMIT,
                  "relations": sorted(relations), "relation_count": len(relations),
                  "cap_per_relation": cap}


def verify_semeval(root: str | Path) -> dict:
    root = Path(root)
    counts = {}
    for name, expected in SEMEVAL_HASHES.items():
        path = root / name
        if sha(path) != expected:
            raise ValueError(f"SemEval hash mismatch: {name}")
        if name in SEMEVAL_ROWS:
            # Counting bytes does not parse or expose the sealed test records.
            count = sum(1 for line in path.open("rb") if line.strip())
            if count != SEMEVAL_ROWS[name]:
                raise ValueError(f"SemEval row count mismatch: {name}")
            counts[name] = count
    return counts


def load_semeval(root: str | Path, filename: str, split: str) -> list[dict]:
    if filename == "semeval_test.txt":
        raise ValueError("S1 builder may not parse sealed SemEval test")
    rows = []
    for index, line in enumerate((Path(root) / filename).read_text().splitlines()):
        raw = json.loads(line)
        if raw["relation"] == "Other":
            continue
        head, tail = tuple(raw["h"]["pos"]), tuple(raw["t"]["pos"])
        tokens = raw["token"]
        head_text = " ".join(tokens[head[0]:head[1]])
        tail_text = " ".join(tokens[tail[0]:tail[1]])
        rows.append({
            "source": "semeval", "source_split": split,
            "relation": raw["relation"], "source_index": index,
            "source_row_sha256": canonical_sha(raw), "tokens": tokens,
            "head": head, "tail": tail, "head_text": head_text,
            "tail_text": tail_text,
            "head_id": f"{split}:{index}:h", "tail_id": f"{split}:{index}:t",
            "text": render(tokens, head, tail),
        })
    return rows


def filter_eligible_relations(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Remove classes that cannot support both natural positive donor types."""
    by_relation = defaultdict(list)
    for row in rows:
        by_relation[row["relation"]].append(row)
    eligible, excluded = [], []
    for relation, values in sorted(by_relation.items()):
        heads = {normalize(row["head_text"]) for row in values}
        tails = {normalize(row["tail_text"]) for row in values}
        if len(heads) >= 2 and len(tails) >= 2:
            eligible.extend(values)
        else:
            excluded.extend({"source": row["source"], "source_split": row["source_split"],
                             "relation": relation,
                             "source_row_sha256": row["source_row_sha256"],
                             "reason": "undercovered_relation_for_structured_pair"}
                            for row in values)
    return eligible, excluded


def _different(values: list[dict], anchor: dict, field: str, salt: str, seed: int) -> dict:
    candidates = [row for row in values if normalize(row[field]) != normalize(anchor[field])]
    if not candidates:
        raise ValueError(f"no different {field} donor")
    return min(candidates, key=lambda row: stable(seed, salt, anchor["source_row_sha256"],
                                                  row["source_row_sha256"]))


def _lexical_score(left: str, right: str) -> float:
    def grams(text):
        text = f"  {normalize(text)}  "
        return {text[index:index + 3] for index in range(max(1, len(text) - 2))}
    a, b = grams(left), grams(right)
    return len(a & b) / max(1, len(a | b))


def _hard_other_relation(rows: list[dict], anchor: dict, seed: int) -> dict:
    pool = _cyclic_candidates(
        rows, anchor, lambda row: row["relation"] != anchor["relation"],
        "hard-pool", 96, seed,
    )
    return max(pool, key=lambda row: (_lexical_score(anchor["text"], row["text"]),
                                      stable(seed, "hard-tie", row["source_row_sha256"])))


def _cyclic_candidates(rows: list[dict], anchor: dict, predicate, salt: str,
                       limit: int, seed: int) -> list[dict]:
    start = int(stable(seed, salt, anchor["source_row_sha256"])[:16], 16) % len(rows)
    result = []
    for offset in range(len(rows)):
        row = rows[(start + offset) % len(rows)]
        if row["source_row_sha256"] != anchor["source_row_sha256"] and predicate(row):
            result.append(row)
            if len(result) == limit:
                break
    if not result:
        raise ValueError(f"no candidates for {salt}")
    return result


def operation(entity: bool, relation: bool, value: bool) -> str:
    if entity and relation:
        return "duplicate" if value else "update"
    return "retain"


def _pair(anchor: dict, right: dict, kind: str, right_text: str,
          same_entity: bool, same_relation: bool, same_value: bool,
          split: str, seed: int) -> dict:
    pair_id = stable(PROTOCOL, seed, split, kind, anchor["source_row_sha256"],
                     right["source_row_sha256"])[:24]
    op = operation(same_entity, same_relation, same_value)
    return {
        "pair_id": pair_id, "split": split, "source": anchor["source"], "kind": kind,
        "left_text": anchor["text"], "right_text": right_text,
        "same_entity": int(same_entity), "same_relation": int(same_relation),
        "same_value": int(same_value), "operation": op,
        "left_source_row_sha256": anchor["source_row_sha256"],
        "right_source_row_sha256": right["source_row_sha256"],
        "left_relation_audit": anchor["relation"],
        "right_relation_audit": right["relation"],
    }


def construct_pairs(rows: list[dict], split: str, seed: int = SEED) -> list[dict]:
    rows = sorted(rows, key=lambda row: row["source_row_sha256"])
    by_relation = defaultdict(list)
    for row in rows:
        by_relation[row["relation"]].append(row)
    if len(by_relation) < 2:
        raise ValueError("structured corpus needs at least two relations")
    output = []
    for anchor in rows:
        peers = [row for row in by_relation[anchor["relation"]]
                 if row["source_row_sha256"] != anchor["source_row_sha256"]]
        relation_peer = _different(peers, anchor, "head_text", "relation-peer", seed)
        value_donor = _different(peers, anchor, "tail_text", "value-donor", seed)
        hard = _hard_other_relation(rows, anchor, seed)
        unrelated = _cyclic_candidates(
            rows, anchor,
            lambda row: (row["relation"] != anchor["relation"]
                         and normalize(row["head_text"]) != normalize(anchor["head_text"])
                         and normalize(row["tail_text"]) != normalize(anchor["tail_text"])),
            "unrelated", 1, seed,
        )[0]

        output.append(_pair(anchor, anchor, "duplicate", anchor["text"], True, True, True,
                            split, seed))
        updated_text = render(anchor["tokens"], anchor["head"], anchor["tail"],
                              tail_value=value_donor["tail_text"])
        output.append(_pair(anchor, value_donor, "update", updated_text, True, True, False,
                            split, seed))
        output.append(_pair(anchor, relation_peer, "same_relation_other_entity",
                            relation_peer["text"], False, True,
                            normalize(anchor["tail_text"]) == normalize(relation_peer["tail_text"]),
                            split, seed))
        hard_text = render(hard["tokens"], hard["head"], hard["tail"],
                           head_value=anchor["head_text"])
        output.append(_pair(anchor, hard, "same_entity_other_relation", hard_text, True, False,
                            normalize(anchor["tail_text"]) == normalize(hard["tail_text"]),
                            split, seed))
        output.append(_pair(anchor, unrelated, "unrelated", unrelated["text"], False, False,
                            False, split, seed))
    output.sort(key=lambda row: row["pair_id"])
    if len({row["pair_id"] for row in output}) != len(output):
        raise RuntimeError("pair ID collision")
    return output


def summarize(rows: list[dict]) -> dict:
    return {
        "pairs": len(rows), "anchors": len(rows) // len(KINDS),
        "by_kind": dict(sorted(Counter(row["kind"] for row in rows).items())),
        "by_operation": dict(sorted(Counter(row["operation"] for row in rows).items())),
        "relations": len({row["left_relation_audit"] for row in rows}),
    }


def build(fewrel_root: str | Path, semeval_root: str | Path, output: str | Path,
          fewrel_cap: int = 200, seed: int = SEED) -> dict:
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    semeval_counts = verify_semeval(semeval_root)
    train_instances, fewrel_meta = load_fewrel_train(fewrel_root, fewrel_cap, seed)
    calibration_instances = load_semeval(semeval_root, "semeval_train.txt", "calibration")
    development_instances = load_semeval(semeval_root, "semeval_val.txt", "development")
    exclusions = {}
    train_instances, exclusions["train"] = filter_eligible_relations(train_instances)
    calibration_instances, exclusions["calibration"] = filter_eligible_relations(
        calibration_instances)
    development_instances, exclusions["development"] = filter_eligible_relations(
        development_instances)
    split_pairs = {
        "train": construct_pairs(train_instances, "train", seed),
        "calibration": construct_pairs(calibration_instances, "calibration", seed),
        "development": construct_pairs(development_instances, "development", seed),
    }
    output.mkdir(parents=True)
    files = {}
    for split, rows in split_pairs.items():
        path = output / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
                                for row in rows))
        files[path.name] = sha(path)
    manifest = {
        "protocol": PROTOCOL, "seed": seed, "fewrel": fewrel_meta,
        "semeval_source_sha256": SEMEVAL_HASHES,
        "semeval_source_rows": semeval_counts,
        "counts": {split: summarize(rows) for split, rows in split_pairs.items()},
        "eligibility_exclusions": {split: values for split, values in exclusions.items()},
        "eligibility_exclusion_counts": {split: len(values)
                                         for split, values in exclusions.items()},
        "files_sha256": files, "kinds": KINDS,
        "fixed_operation_rule": True,
        "semeval_test_parsed": False, "semeval_test_examples_written": False,
        "fewrel_nontrain_examples_written": False,
        "model_outputs_loaded": False, "terminal_status": "built_before_model_execution",
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
