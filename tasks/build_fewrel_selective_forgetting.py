"""Build the preregistered subject- and relation-disjoint FewRel pair corpus."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "fewrel_selective_forgetting_pairs_v1"
SOURCE_COMMIT = "278a2315d2138810a379cd8d5718914dc56e2582"
SOURCE_HASHES = {
    "train_wiki.json": "8e9e58a05d559be773cbe18a8d4a16ba17e6d397663f9f05a056f894a94b92c1",
    "val_wiki.json": "76bb44c2485b6faed358a3ce040eadafcc8a8835245cd7e6afc1bef672d4a0bf",
    "pid2name.json": "eeedd25512a8e4c2bc06b78009836d27de81807bcba6ed929debd5cd713c07ea",
}
SEED = 2311
RELATION_CAP = 300
MIN_RELATION_ROWS = 250
HARD_CANDIDATES = 128
PARTITIONS = ("train", "development", "internal_test", "external_test")


def stable(*values):
    return hashlib.sha256(":.:".join(map(str, values)).encode()).hexdigest()


def normalize(value):
    return " ".join(str(value).casefold().split())


def canonical_sha(value):
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def relation_partitions(train_relations, external_relations, seed=SEED):
    external_relations = tuple(external_relations)
    ordered = sorted(train_relations,
                     key=lambda value: stable(seed, "relation", value))
    if len(ordered) != 64 or len(external_relations) != 16:
        raise ValueError("unexpected FewRel relation inventory")
    result = {
        "train": ordered[:48],
        "development": ordered[48:56],
        "internal_test": ordered[56:],
        "external_test": sorted(external_relations),
    }
    if len(set().union(*map(set, result.values()))) != 80:
        raise ValueError("FewRel relation partitions overlap")
    return result


def span_indices(entity):
    spans = entity[2]
    if not spans or any(not span for span in spans):
        raise ValueError("empty FewRel entity span")
    return [tuple(int(index) for index in span) for span in spans]


def masked_surface(row):
    tokens = row["tokens"]
    heads, tails = span_indices(row["h"]), span_indices(row["t"])
    head_positions = {index for span in heads for index in span}
    tail_positions = {index for span in tails for index in span}
    if head_positions & tail_positions:
        raise ValueError("overlapping FewRel head/tail spans")
    if any(index < 0 or index >= len(tokens)
           for index in head_positions | tail_positions):
        raise ValueError("FewRel entity span is out of range")
    starts = {}
    covered = set()
    for marker, spans in (("[SUBJECT]", heads), ("[VALUE]", tails)):
        for span in spans:
            ordered = sorted(span)
            if ordered != list(range(ordered[0], ordered[-1] + 1)):
                raise ValueError("noncontiguous FewRel entity span")
            starts[ordered[0]] = marker
            covered.update(ordered)
    output = []
    for index, token in enumerate(tokens):
        if index in starts:
            output.append(starts[index])
        if index not in covered:
            output.append(str(token))
    text = " ".join(output)
    if text.count("[SUBJECT]") < 1 or text.count("[VALUE]") < 1:
        raise ValueError("masked FewRel surface lost an entity marker")
    return " ".join(text.split())


def trigrams(value):
    value = f"  {normalize(value)}  "
    return {value[index:index + 3] for index in range(max(0, len(value) - 2))}


def trigram_jaccard(left, right):
    left, right = trigrams(left), trigrams(right)
    return len(left & right) / len(left | right) if left or right else 1.0


def verify_source(source_root):
    source_root = Path(source_root)
    observed = {name: sha(source_root / "data" / name) for name in SOURCE_HASHES}
    if observed != SOURCE_HASHES:
        raise ValueError(f"FewRel source hash mismatch: {observed}")
    head = (source_root / ".git" / "HEAD").read_text().strip()
    if head.startswith("ref:"):
        ref = source_root / ".git" / head.split(maxsplit=1)[1]
        commit = ref.read_text().strip()
    else:
        commit = head
    if commit != SOURCE_COMMIT:
        raise ValueError(f"FewRel source commit mismatch: {commit}")
    return observed


def raw_inventory(train, external, partitions):
    output = []
    for partition in PARTITIONS:
        source = external if partition == "external_test" else train
        source_file = "val_wiki.json" if partition == "external_test" else "train_wiki.json"
        for relation in partitions[partition]:
            for index, row in enumerate(source[relation]):
                output.append({
                    "partition": partition, "source_file": source_file,
                    "relation_id": relation, "source_index": index,
                    "head_entity_id": row["h"][1], "tail_entity_id": row["t"][1],
                    "source_row_sha256": canonical_sha(row), "raw": row,
                })
    return output


def filter_instances(raw_rows, relation_names, seed=SEED, cap=RELATION_CAP):
    exclusions = []
    head_owners = defaultdict(set)
    for row in raw_rows:
        head_owners[row["head_entity_id"]].add(row["partition"])
    candidates = []
    for row in raw_rows:
        if len(head_owners[row["head_entity_id"]]) != 1:
            exclusions.append({**{key: row[key] for key in (
                "partition", "source_file", "relation_id", "source_index",
                "source_row_sha256")}, "reason": "head_entity_cross_partition"})
            continue
        surface = masked_surface(row["raw"])
        candidates.append({
            **{key: row[key] for key in (
                "partition", "source_file", "relation_id", "source_index",
                "head_entity_id", "tail_entity_id", "source_row_sha256")},
            "relation_name": relation_names[row["relation_id"]][0],
            "surface": surface, "surface_normalized": normalize(surface),
        })

    surface_partitions, surface_relations = defaultdict(set), defaultdict(set)
    for row in candidates:
        surface_partitions[row["surface_normalized"]].add(row["partition"])
        surface_relations[(row["partition"], row["surface_normalized"])].add(
            row["relation_id"])
    eligible = []
    for row in candidates:
        reason = None
        if len(surface_partitions[row["surface_normalized"]]) != 1:
            reason = "surface_cross_partition"
        elif len(surface_relations[(row["partition"], row["surface_normalized"])]) != 1:
            reason = "surface_multi_relation"
        if reason:
            exclusions.append({**{key: row[key] for key in (
                "partition", "source_file", "relation_id", "source_index",
                "source_row_sha256")}, "reason": reason})
        else:
            eligible.append(row)

    groups = defaultdict(list)
    for row in eligible:
        groups[(row["partition"], row["relation_id"],
                row["surface_normalized"])].append(row)
    deduplicated = []
    for values in groups.values():
        ordered = sorted(values, key=lambda row: stable(
            seed, "surface-dedup", row["source_row_sha256"]))
        deduplicated.append(ordered[0])
        for row in ordered[1:]:
            exclusions.append({**{key: row[key] for key in (
                "partition", "source_file", "relation_id", "source_index",
                "source_row_sha256")}, "reason": "duplicate_surface_within_relation"})

    by_relation = defaultdict(list)
    for row in deduplicated:
        by_relation[(row["partition"], row["relation_id"])].append(row)
    retained = []
    for (partition, relation), values in sorted(by_relation.items()):
        ordered = sorted(values, key=lambda row: stable(
            seed, "relation-cap", partition, relation, row["source_row_sha256"]))
        retained.extend(ordered[:cap])
        for row in ordered[cap:]:
            exclusions.append({**{key: row[key] for key in (
                "partition", "source_file", "relation_id", "source_index",
                "source_row_sha256")}, "reason": "relation_cap"})
    retained.sort(key=lambda row: (PARTITIONS.index(row["partition"]),
                                   row["relation_id"], row["source_row_sha256"]))
    for row in retained:
        row["instance_id"] = stable(
            seed, row["partition"], row["relation_id"], row["source_row_sha256"])[:24]
        row.pop("surface_normalized")
    return retained, sorted(exclusions, key=lambda row: (
        PARTITIONS.index(row["partition"]), row["relation_id"], row["source_index"],
        row["reason"]))


def pair_record(kind, anchor, other, label, seed=SEED):
    pair_id = stable(seed, "pair", kind, anchor["instance_id"],
                     other["instance_id"])[:24]
    return {
        "pair_id": pair_id, "partition": anchor["partition"], "kind": kind,
        "left_instance_id": anchor["instance_id"],
        "right_instance_id": other["instance_id"],
        "left_relation_id": anchor["relation_id"],
        "right_relation_id": other["relation_id"],
        "left_head_entity_id": anchor["head_entity_id"],
        "right_head_entity_id": other["head_entity_id"],
        "left_surface": anchor["surface"], "right_surface": other["surface"],
        "same_relation": int(label), "locality_risk": int(not label),
        "character_trigram_jaccard": trigram_jaccard(
            anchor["surface"], other["surface"]),
    }


def construct_partition_pairs(instances, partition, seed=SEED):
    rows = [row for row in instances if row["partition"] == partition]
    by_relation, by_head, by_trigram = defaultdict(list), defaultdict(list), defaultdict(set)
    for index, row in enumerate(rows):
        by_relation[row["relation_id"]].append(index)
        by_head[row["head_entity_id"]].append(index)
        for trigram in trigrams(row["surface"]):
            by_trigram[trigram].add(index)
    order = sorted(range(len(rows)), key=lambda index: rows[index]["source_row_sha256"])
    pairs = []
    for anchor_index, anchor in enumerate(rows):
        positives = [index for index in by_relation[anchor["relation_id"]]
                     if index != anchor_index
                     and rows[index]["head_entity_id"] != anchor["head_entity_id"]]
        positives.sort(key=lambda index: stable(
            seed, "positive", anchor["instance_id"], rows[index]["instance_id"]))
        if len(positives) < 2:
            raise ValueError(f"insufficient positive peers for {anchor['instance_id']}")
        for index in positives[:2]:
            pairs.append(pair_record("same_relation_positive", anchor, rows[index], 1, seed))

        same_subject = [index for index in by_head[anchor["head_entity_id"]]
                        if rows[index]["relation_id"] != anchor["relation_id"]]
        if same_subject:
            hard_index = max(same_subject, key=lambda index: (
                trigram_jaccard(anchor["surface"], rows[index]["surface"]),
                stable(seed, "same-subject-tie", rows[index]["instance_id"])))
            hard_kind = "same_subject_hard_negative"
        else:
            lexical = set()
            for trigram in trigrams(anchor["surface"]):
                lexical.update(by_trigram[trigram])
            lexical = [index for index in lexical
                       if rows[index]["relation_id"] != anchor["relation_id"]]
            lexical.sort(key=lambda index: stable(
                seed, "hard-pool", anchor["instance_id"], rows[index]["instance_id"]))
            lexical = lexical[:HARD_CANDIDATES]
            if not lexical:
                lexical = [index for index in order
                           if rows[index]["relation_id"] != anchor["relation_id"]]
            hard_index = max(lexical, key=lambda index: (
                trigram_jaccard(anchor["surface"], rows[index]["surface"]),
                stable(seed, "lexical-tie", rows[index]["instance_id"])))
            hard_kind = "lexical_hard_negative"
        pairs.append(pair_record(hard_kind, anchor, rows[hard_index], 0, seed))

        start = int(stable(seed, "random", anchor["instance_id"])[:12], 16) % len(order)
        random_index = None
        for offset in range(len(order)):
            index = order[(start + offset) % len(order)]
            if (rows[index]["relation_id"] != anchor["relation_id"]
                    and index != hard_index):
                random_index = index
                break
        if random_index is None:
            raise ValueError("unable to construct FewRel random negative")
        pairs.append(pair_record("random_negative", anchor, rows[random_index], 0, seed))

    unique = {}
    for row in pairs:
        identity = (row["kind"],) + tuple(sorted((row["left_instance_id"],
                                                   row["right_instance_id"])))
        unique.setdefault(identity, row)
    positives = [row for row in unique.values() if row["same_relation"]]
    negatives = [row for row in unique.values() if not row["same_relation"]]
    keep = min(len(positives), len(negatives))
    positives = sorted(positives, key=lambda row: stable(seed, "balance", row["pair_id"]))[:keep]
    negatives = sorted(negatives, key=lambda row: stable(seed, "balance", row["pair_id"]))[:keep]
    return sorted(positives + negatives, key=lambda row: row["pair_id"])


def corpus_counts(instances, pairs, exclusions, partitions):
    output = {}
    for partition in PARTITIONS:
        rows = [row for row in instances if row["partition"] == partition]
        values = [row for row in pairs if row["partition"] == partition]
        output[partition] = {
            "relations": len(partitions[partition]), "instances": len(rows),
            "minimum_instances_per_relation": min(Counter(
                row["relation_id"] for row in rows).values()),
            "pairs": len(values), "positive": sum(row["same_relation"] for row in values),
            "negative": sum(not row["same_relation"] for row in values),
            "by_kind": dict(sorted(Counter(row["kind"] for row in values).items())),
        }
    return {"by_partition": output, "instances": len(instances), "pairs": len(pairs),
            "exclusions": len(exclusions),
            "exclusions_by_reason": dict(sorted(Counter(
                row["reason"] for row in exclusions).items()))}


def build(source_root, output, seed=SEED, relation_cap=RELATION_CAP):
    source_root, output = Path(source_root), Path(output)
    hashes = verify_source(source_root)
    train = json.loads((source_root / "data/train_wiki.json").read_text())
    external = json.loads((source_root / "data/val_wiki.json").read_text())
    names = json.loads((source_root / "data/pid2name.json").read_text())
    partitions = relation_partitions(train, external, seed)
    raw_rows = raw_inventory(train, external, partitions)
    instances, exclusions = filter_instances(raw_rows, names, seed, relation_cap)
    per_relation = Counter((row["partition"], row["relation_id"]) for row in instances)
    if set(per_relation) != {(partition, relation) for partition, relations in partitions.items()
                             for relation in relations}:
        raise RuntimeError("FewRel relation lost all instances")
    if min(per_relation.values()) < MIN_RELATION_ROWS:
        raise RuntimeError("FewRel minimum retained relation coverage failed")
    pairs = []
    for partition in PARTITIONS:
        pairs.extend(construct_partition_pairs(instances, partition, seed))
    counts = corpus_counts(instances, pairs, exclusions, partitions)
    for value in counts["by_partition"].values():
        if value["positive"] != value["negative"]:
            raise RuntimeError("FewRel pair partition is not balanced")
        if not {"same_relation_positive", "random_negative"} <= set(value["by_kind"]):
            raise RuntimeError("FewRel pair kind coverage failed")
        if not ({"same_subject_hard_negative", "lexical_hard_negative"}
                & set(value["by_kind"])):
            raise RuntimeError("FewRel hard-negative coverage failed")

    output.mkdir(parents=True, exist_ok=False)
    file_rows = {"instances.jsonl": instances, "pairs.jsonl": pairs,
                 "exclusions.jsonl": exclusions}
    file_hashes = {}
    for name, values in file_rows.items():
        path = output / name
        path.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
                                for row in values))
        file_hashes[name] = sha(path)
    manifest = {
        "protocol": PROTOCOL, "seed": seed, "relation_cap": relation_cap,
        "minimum_relation_rows": MIN_RELATION_ROWS,
        "hard_candidate_pool": HARD_CANDIDATES,
        "source": {"repository_commit": SOURCE_COMMIT, "files_sha256": hashes},
        "partitions": partitions, "counts": counts, "files_sha256": file_hashes,
        "relation_ids_are_supervision_and_audit_only": True,
        "head_entities_disjoint_across_partitions": True,
        "tail_entities_masked_but_not_required_disjoint": True,
        "model_outputs_loaded": False, "features_extracted": False,
        "internal_test_scored": False, "external_test_scored": False,
        "sealed_benchmarks_loaded": False,
        "terminal_status": "frozen_fewrel_pairs_before_feature_extraction",
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--relation-cap", type=int, default=RELATION_CAP)
    print(json.dumps(build(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
