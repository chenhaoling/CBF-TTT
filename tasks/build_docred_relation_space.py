"""Build the preregistered relation-disjoint DocRED semantic-slot corpus."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path

from tasks.build_fewrel_selective_forgetting import (
    canonical_sha, load_jsonl, normalize, stable, trigram_jaccard, trigrams,
)
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "docred_relation_description_space_v1"
SOURCE_HASHES = {
    "train_annotated.json.gz":
        "0d01cd07cabd7f9077db6ea8628832cf60281b4228ec1ba54647f836a3b17d02",
    "dev.json.gz":
        "6ae4d7f5b0b9d2cbe74b9634ed43b35b7cb5b7c0dc3a16226dbe343139a4ae05",
    "rel_info.json.gz":
        "7ef27efff537ba89ae66f6f4e60e4908d4df3860a4c2819ea94fa5ed696bdc70",
}
FEWREL_MANIFEST_SHA256 = (
    "6f1dc9c9bacc934c5e44693b59a841220e18772ef0250eb64a5afb76aee77a1d"
)
SEED = 2351
TRAIN_MIN = 50
DEV_MIN = 20
HELDOUT_RETAINED_MIN = 15
CAPS = {"representation_train": 512, "relation_development": 128,
        "relation_source_test": 128}
PARTITIONS = tuple(CAPS)
HARD_CANDIDATES = 128


def load_gzip_json(path):
    with gzip.open(path, "rt", encoding="utf-8") as source:
        return json.load(source)


def evidence_counts(documents):
    return Counter(label["r"] for document in documents
                   for label in document.get("labels", []) if label.get("evidence"))


def relation_partitions(train, dev, relation_info, excluded, seed=SEED):
    train_counts, dev_counts = evidence_counts(train), evidence_counts(dev)
    eligible = [relation for relation in relation_info
                if relation not in excluded
                and train_counts[relation] >= TRAIN_MIN
                and dev_counts[relation] >= DEV_MIN]
    ordered = sorted(eligible, key=lambda value: hashlib.sha256(
        f"{seed}:relation:{value}".encode()).hexdigest())
    if len(ordered) != 67:
        raise ValueError(f"unexpected eligible DocRED relation count: {len(ordered)}")
    return {"representation_train": ordered[:51],
            "relation_development": ordered[51:59],
            "relation_source_test": ordered[59:]}, train_counts, dev_counts


def mention_sentences(document, entity_index):
    return sorted({int(mention["sent_id"])
                   for mention in document["vertexSet"][entity_index]})


def selected_sentences(document, label, maximum=4):
    evidence = sorted(set(map(int, label["evidence"])))
    head_sentences = mention_sentences(document, int(label["h"]))
    tail_sentences = mention_sentences(document, int(label["t"]))
    if not head_sentences or not tail_sentences:
        raise ValueError("DocRED relation entity has no mention")
    selected = list(evidence)
    if not set(selected) & set(head_sentences):
        selected.append(head_sentences[0])
    if not set(selected) & set(tail_sentences):
        selected.append(tail_sentences[0])
    selected = sorted(set(selected))
    if len(selected) > maximum:
        mandatory = {next(value for value in selected if value in head_sentences),
                     next(value for value in selected if value in tail_sentences)}
        retained = sorted(mandatory)
        for sentence in evidence:
            if sentence not in retained and len(retained) < maximum:
                retained.append(sentence)
        selected = sorted(retained[:maximum])
    return selected


def mask_sentence(document, sentence_id, head, tail):
    tokens = document["sents"][sentence_id]
    starts, covered = {}, set()
    for marker, entity in (("[SUBJECT]", head), ("[VALUE]", tail)):
        for mention in document["vertexSet"][entity]:
            if int(mention["sent_id"]) != sentence_id:
                continue
            start, end = map(int, mention["pos"])
            if not 0 <= start < end <= len(tokens):
                raise ValueError("DocRED mention span out of range")
            positions = set(range(start, end))
            if covered & positions:
                raise ValueError("overlapping DocRED relation mentions")
            starts[start] = marker; covered.update(positions)
    output = []
    for index, token in enumerate(tokens):
        if index in starts:
            output.append(starts[index])
        if index not in covered:
            output.append(str(token))
    return " ".join(output)


def masked_evidence(document, label):
    sentences = selected_sentences(document, label)
    text = " [SEP] ".join(mask_sentence(
        document, sentence, int(label["h"]), int(label["t"]))
        for sentence in sentences)
    text = " ".join(text.split())
    if "[SUBJECT]" not in text or "[VALUE]" not in text:
        raise ValueError("DocRED evidence lost a relation entity marker")
    return text, sentences


def raw_candidates(train, dev, partitions, relation_info):
    relation_to_partition = {relation: partition
                             for partition, values in partitions.items()
                             for relation in values}
    output, exclusions = [], []
    for source_file, documents, allowed_partition in (
            ("train_annotated.json.gz", train, "representation_train"),
            ("dev.json.gz", dev, None)):
        document_owners = {}
        if allowed_partition is None:
            for document_index, document in enumerate(documents):
                owners = {relation_to_partition[label["r"]]
                          for label in document.get("labels", [])
                          if label.get("evidence") and label["r"] in relation_to_partition
                          and relation_to_partition[label["r"]] != "representation_train"}
                document_owners[document_index] = owners
        for document_index, document in enumerate(documents):
            for label_index, label in enumerate(document.get("labels", [])):
                relation = label["r"]
                if not label.get("evidence") or relation not in relation_to_partition:
                    continue
                partition = relation_to_partition[relation]
                if allowed_partition is not None and partition != allowed_partition:
                    continue
                identity = {"partition": partition, "source_file": source_file,
                            "document_index": document_index, "label_index": label_index,
                            "relation_id": relation}
                if allowed_partition is None and document_owners[document_index] != {partition}:
                    exclusions.append({**identity, "reason": "heldout_document_cross_partition"})
                    continue
                surface, sentence_ids = masked_evidence(document, label)
                output.append({
                    **identity, "document_id": stable(
                        SEED, source_file, document_index, document.get("title", ""))[:24],
                    "document_title": document.get("title", ""),
                    "source_document_sha256": canonical_sha(document),
                    "source_label_sha256": canonical_sha(label),
                    "head_entity_index": int(label["h"]),
                    "tail_entity_index": int(label["t"]),
                    "selected_sentence_ids": sentence_ids, "surface": surface,
                    "surface_normalized": normalize(surface),
                })
    return output, exclusions


def filter_instances(candidates, exclusions, seed=SEED):
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
            exclusions.append({key: value for key, value in row.items()
                               if key in ("partition", "source_file", "document_index",
                                          "label_index", "relation_id")} | {"reason": reason})
        else:
            eligible.append(row)
    groups = defaultdict(list)
    for row in eligible:
        groups[(row["partition"], row["relation_id"],
                row["surface_normalized"])].append(row)
    deduplicated = []
    for values in groups.values():
        ordered = sorted(values, key=lambda row: stable(
            seed, "surface-dedup", row["source_label_sha256"]))
        deduplicated.append(ordered[0])
        for row in ordered[1:]:
            exclusions.append({key: value for key, value in row.items()
                               if key in ("partition", "source_file", "document_index",
                                          "label_index", "relation_id")} |
                              {"reason": "duplicate_surface_within_relation"})
    by_relation = defaultdict(list)
    for row in deduplicated:
        by_relation[(row["partition"], row["relation_id"])].append(row)
    retained = []
    for (partition, relation), values in sorted(by_relation.items()):
        ordered = sorted(values, key=lambda row: stable(
            seed, "relation-cap", partition, relation, row["source_label_sha256"]))
        retained.extend(ordered[:CAPS[partition]])
        for row in ordered[CAPS[partition]:]:
            exclusions.append({key: value for key, value in row.items()
                               if key in ("partition", "source_file", "document_index",
                                          "label_index", "relation_id")} |
                              {"reason": "relation_cap"})
    retained.sort(key=lambda row: (PARTITIONS.index(row["partition"]),
                                   row["relation_id"], row["source_label_sha256"]))
    for row in retained:
        row["instance_id"] = stable(
            seed, row["partition"], row["source_file"], row["document_index"],
            row["label_index"], row["source_label_sha256"])[:24]
        row.pop("surface_normalized")
    exclusions.sort(key=lambda row: (
        PARTITIONS.index(row["partition"]), row["source_file"],
        row["document_index"], row["label_index"], row["reason"]))
    return retained, exclusions


def pair_record(kind, anchor, other, label, seed=SEED):
    return {
        "pair_id": stable(seed, "docred-pair", kind, anchor["instance_id"],
                          other["instance_id"])[:24],
        "partition": anchor["partition"], "kind": kind,
        "left_instance_id": anchor["instance_id"],
        "right_instance_id": other["instance_id"],
        "left_relation_id": anchor["relation_id"],
        "right_relation_id": other["relation_id"],
        "left_document_id": anchor["document_id"],
        "right_document_id": other["document_id"],
        "same_relation": int(label), "locality_risk": int(not label),
        "character_trigram_jaccard": trigram_jaccard(
            anchor["surface"], other["surface"]),
    }


def construct_pairs(instances, partition, seed=SEED):
    rows = [row for row in instances if row["partition"] == partition]
    by_relation, by_trigram = defaultdict(list), defaultdict(set)
    for index, row in enumerate(rows):
        by_relation[row["relation_id"]].append(index)
        for value in trigrams(row["surface"]):
            by_trigram[value].add(index)
    order = sorted(range(len(rows)), key=lambda index: rows[index]["source_label_sha256"])
    pairs = []
    for anchor_index, anchor in enumerate(rows):
        positives = [index for index in by_relation[anchor["relation_id"]]
                     if index != anchor_index
                     and rows[index]["document_id"] != anchor["document_id"]]
        positives.sort(key=lambda index: stable(
            seed, "positive", anchor["instance_id"], rows[index]["instance_id"]))
        if len(positives) < 2:
            raise ValueError(f"insufficient DocRED positive peers: {anchor['instance_id']}")
        for index in positives[:2]:
            pairs.append(pair_record("same_relation_positive", anchor, rows[index], 1, seed))
        lexical = set()
        for value in trigrams(anchor["surface"]):
            lexical.update(by_trigram[value])
        lexical = [index for index in lexical
                   if rows[index]["relation_id"] != anchor["relation_id"]]
        lexical.sort(key=lambda index: stable(
            seed, "hard-pool", anchor["instance_id"], rows[index]["instance_id"]))
        lexical = lexical[:HARD_CANDIDATES]
        if not lexical:
            lexical = [index for index in order
                       if rows[index]["relation_id"] != anchor["relation_id"]]
        hard = max(lexical, key=lambda index: (
            trigram_jaccard(anchor["surface"], rows[index]["surface"]),
            stable(seed, "hard-tie", rows[index]["instance_id"])))
        pairs.append(pair_record("lexical_hard_negative", anchor, rows[hard], 0, seed))
        start = int(stable(seed, "random", anchor["instance_id"])[:12], 16) % len(order)
        random_index = None
        for offset in range(len(order)):
            index = order[(start + offset) % len(order)]
            if rows[index]["relation_id"] != anchor["relation_id"] and index != hard:
                random_index = index; break
        if random_index is None:
            raise ValueError("unable to construct DocRED random negative")
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


def counts(instances, pairs, exclusions, partitions):
    result = {}
    for partition in PARTITIONS:
        rows = [row for row in instances if row["partition"] == partition]
        values = [row for row in pairs if row["partition"] == partition]
        relation_counts = Counter(row["relation_id"] for row in rows)
        result[partition] = {
            "relations": len(partitions[partition]), "instances": len(rows),
            "minimum_instances_per_relation": min(relation_counts.values()),
            "maximum_instances_per_relation": max(relation_counts.values()),
            "documents": len({row["document_id"] for row in rows}),
            "pairs": len(values), "positive": sum(row["same_relation"] for row in values),
            "negative": sum(not row["same_relation"] for row in values),
            "by_kind": dict(sorted(Counter(row["kind"] for row in values).items())),
        }
    return {"instances": len(instances), "pairs": len(pairs),
            "exclusions": len(exclusions), "by_partition": result,
            "exclusions_by_reason": dict(sorted(Counter(
                row["reason"] for row in exclusions).items()))}


def build(source_root, fewrel_root, output):
    source_root, fewrel_root, output = map(Path, (source_root, fewrel_root, output))
    observed = {name: sha(source_root / name) for name in SOURCE_HASHES}
    if observed != SOURCE_HASHES:
        raise ValueError("DocRED source hashes changed")
    if sha(fewrel_root / "data/manifest.json") != FEWREL_MANIFEST_SHA256:
        raise ValueError("FewRel sealed relation manifest changed")
    fewrel = json.loads((fewrel_root / "data/manifest.json").read_text())
    excluded = set(fewrel["partitions"]["internal_test"]
                   + fewrel["partitions"]["external_test"])
    train = load_gzip_json(source_root / "train_annotated.json.gz")
    dev = load_gzip_json(source_root / "dev.json.gz")
    relation_info = load_gzip_json(source_root / "rel_info.json.gz")
    partitions, train_counts, dev_counts = relation_partitions(
        train, dev, relation_info, excluded)
    candidates, exclusions = raw_candidates(train, dev, partitions, relation_info)
    instances, exclusions = filter_instances(candidates, exclusions)
    relation_counts = Counter((row["partition"], row["relation_id"]) for row in instances)
    if set(relation_counts) != {(partition, relation) for partition, relations
                               in partitions.items() for relation in relations}:
        raise RuntimeError("DocRED relation lost all evidence instances")
    for (partition, _), value in relation_counts.items():
        floor = TRAIN_MIN if partition == "representation_train" else HELDOUT_RETAINED_MIN
        if value < floor:
            raise RuntimeError(f"DocRED retained relation coverage below {floor}")
    pairs = []
    for partition in PARTITIONS:
        pairs.extend(construct_pairs(instances, partition))
    summary = counts(instances, pairs, exclusions, partitions)
    for value in summary["by_partition"].values():
        if value["positive"] != value["negative"]:
            raise RuntimeError("DocRED pair partition is not balanced")

    descriptions = [{"relation_id": relation,
                     "description": relation_info[relation]}
                    for relation in partitions["representation_train"]]
    output.mkdir(parents=True, exist_ok=False)
    artifacts = {"instances.jsonl": instances, "pairs.jsonl": pairs,
                 "exclusions.jsonl": exclusions,
                 "train_relation_descriptions.jsonl": descriptions}
    hashes = {}
    for name, values in artifacts.items():
        path = output / name
        path.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
                                for row in values))
        hashes[name] = sha(path)
    manifest = {
        "protocol": PROTOCOL, "seed": SEED,
        "source_files_sha256": observed,
        "fewrel_manifest_sha256": FEWREL_MANIFEST_SHA256,
        "excluded_fewrel_test_relation_ids": sorted(excluded),
        "eligibility": {"train_evidence_minimum": TRAIN_MIN,
                        "dev_evidence_minimum": DEV_MIN,
                        "heldout_retained_minimum": HELDOUT_RETAINED_MIN},
        "caps": CAPS, "partitions": partitions,
        "eligible_source_counts": {relation: {
            "train": train_counts[relation], "dev": dev_counts[relation]}
            for values in partitions.values() for relation in values},
        "counts": summary, "files_sha256": hashes,
        "relation_descriptions_released_for_training_relations_only": True,
        "relation_ids_are_supervision_and_audit_only": True,
        "model_outputs_loaded": False, "features_extracted": False,
        "relation_source_test_scored": False,
        "fewrel_internal_test_loaded": False, "fewrel_external_test_loaded": False,
        "sealed_benchmarks_loaded": False,
        "terminal_status": "frozen_docred_relation_space_before_features",
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--fewrel-root", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(build(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
