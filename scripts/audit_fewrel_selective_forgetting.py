"""Independent structural audit of the frozen FewRel selective-forgetting corpus."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from tasks.build_fewrel_selective_forgetting import (
    MIN_RELATION_ROWS, PARTITIONS, PROTOCOL, RELATION_CAP, SOURCE_COMMIT,
    SOURCE_HASHES, canonical_sha, corpus_counts, load_jsonl, masked_surface,
    normalize, stable,
)
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


AUDIT_PROTOCOL = "audit_fewrel_selective_forgetting_pairs_v1"


def repository_commit(root):
    head = (root / ".git" / "HEAD").read_text().strip()
    if not head.startswith("ref:"):
        return head
    return (root / ".git" / head.split(maxsplit=1)[1]).read_text().strip()


def audit(data_root, source_root):
    data_root, source_root = Path(data_root), Path(source_root)
    manifest_path = data_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    instances = load_jsonl(data_root / "instances.jsonl")
    pairs = load_jsonl(data_root / "pairs.jsonl")
    exclusions = load_jsonl(data_root / "exclusions.jsonl")
    train = json.loads((source_root / "data/train_wiki.json").read_text())
    external = json.loads((source_root / "data/val_wiki.json").read_text())
    names = json.loads((source_root / "data/pid2name.json").read_text())

    source_files = {"train_wiki.json": train, "val_wiki.json": external}
    source_identity = {}
    for file_name, source in source_files.items():
        for relation, rows in source.items():
            for index, row in enumerate(rows):
                source_identity[(file_name, relation, index)] = row

    instance_ids = {row["instance_id"]: row for row in instances}
    instance_source_ids = {(row["source_file"], row["relation_id"],
                            int(row["source_index"])) for row in instances}
    exclusion_source_ids = {(row["source_file"], row["relation_id"],
                             int(row["source_index"])) for row in exclusions}
    source_reproduction = all(
        identity in source_identity
        and row["source_row_sha256"] == canonical_sha(source_identity[identity])
        and row["head_entity_id"] == source_identity[identity]["h"][1]
        and row["tail_entity_id"] == source_identity[identity]["t"][1]
        and row["surface"] == masked_surface(source_identity[identity])
        and row["relation_name"] == names[row["relation_id"]][0]
        for row in instances
        for identity in [(row["source_file"], row["relation_id"],
                          int(row["source_index"]))])

    partition_relations = {partition: set(relations)
                           for partition, relations in manifest["partitions"].items()}
    relation_isolation = (
        set().union(*partition_relations.values())
        and sum(map(len, partition_relations.values()))
            == len(set().union(*partition_relations.values())) == 80
        and all(row["relation_id"] in partition_relations[row["partition"]]
                for row in instances))
    heads = defaultdict(set)
    surfaces = defaultdict(set)
    surface_relations = defaultdict(set)
    for row in instances:
        heads[row["head_entity_id"]].add(row["partition"])
        surfaces[normalize(row["surface"])].add(row["partition"])
        surface_relations[(row["partition"], normalize(row["surface"]))].add(
            row["relation_id"])
    disjoint = all(len(value) == 1 for value in heads.values()) \
        and all(len(value) == 1 for value in surfaces.values()) \
        and all(len(value) == 1 for value in surface_relations.values())

    pair_semantics = True
    pair_id_reproduction = True
    identities = set()
    for row in pairs:
        left = instance_ids.get(row["left_instance_id"])
        right = instance_ids.get(row["right_instance_id"])
        if left is None or right is None:
            pair_semantics = False
            continue
        same_relation = left["relation_id"] == right["relation_id"]
        pair_semantics &= all((
            left["partition"] == right["partition"] == row["partition"],
            row["left_surface"] == left["surface"],
            row["right_surface"] == right["surface"],
            row["left_relation_id"] == left["relation_id"],
            row["right_relation_id"] == right["relation_id"],
            row["left_head_entity_id"] == left["head_entity_id"],
            row["right_head_entity_id"] == right["head_entity_id"],
            int(row["same_relation"]) == int(same_relation),
            int(row["locality_risk"]) == int(not same_relation),
            (row["kind"] == "same_relation_positive") == same_relation,
            row["kind"] != "same_subject_hard_negative"
                or left["head_entity_id"] == right["head_entity_id"],
            row["kind"] not in ("lexical_hard_negative", "random_negative")
                or left["relation_id"] != right["relation_id"],
            row["kind"] != "same_relation_positive"
                or left["head_entity_id"] != right["head_entity_id"],
        ))
        pair_id_reproduction &= row["pair_id"] == stable(
            manifest["seed"], "pair", row["kind"], left["instance_id"],
            right["instance_id"])[:24]
        identity = (row["partition"], row["kind"]) + tuple(sorted((
            left["instance_id"], right["instance_id"])))
        if identity in identities:
            pair_semantics = False
        identities.add(identity)

    counts = corpus_counts(instances, pairs, exclusions, manifest["partitions"])
    relation_counts = Counter((row["partition"], row["relation_id"])
                              for row in instances)
    balanced_and_covered = all(
        value["positive"] == value["negative"]
        and value["minimum_instances_per_relation"] >= MIN_RELATION_ROWS
        and {"same_relation_positive", "random_negative"} <= set(value["by_kind"])
        and bool({"same_subject_hard_negative", "lexical_hard_negative"}
                 & set(value["by_kind"]))
        for value in counts["by_partition"].values()) \
        and max(relation_counts.values()) <= RELATION_CAP
    file_hashes = {name: sha(data_root / name)
                   for name in ("instances.jsonl", "pairs.jsonl", "exclusions.jsonl")}

    checks = {
        "protocol_and_frozen_parameters": (
            manifest["protocol"] == PROTOCOL and manifest["seed"] == 2311
            and manifest["relation_cap"] == RELATION_CAP
            and manifest["minimum_relation_rows"] == MIN_RELATION_ROWS),
        "official_source_commit": repository_commit(source_root) == SOURCE_COMMIT,
        "official_source_hashes": (
            {name: sha(source_root / "data" / name) for name in SOURCE_HASHES}
            == SOURCE_HASHES == manifest["source"]["files_sha256"]),
        "complete_source_accounting": (
            not (instance_source_ids & exclusion_source_ids)
            and instance_source_ids | exclusion_source_ids == set(source_identity)),
        "unique_instances_and_source_rows": (
            len(instance_ids) == len(instances)
            and len(instance_source_ids) == len(instances)),
        "all_masked_instances_reproduced_from_source": source_reproduction,
        "relation_partition_isolation": bool(relation_isolation),
        "head_surface_and_ambiguous_surface_isolation": disjoint,
        "pair_references_labels_and_kinds": bool(pair_semantics),
        "pair_ids_reproduced": bool(pair_id_reproduction),
        "unique_pair_ids": len({row["pair_id"] for row in pairs}) == len(pairs),
        "balanced_pair_and_relation_coverage": balanced_and_covered,
        "manifest_counts_recomputed": counts == manifest["counts"],
        "file_hashes_recomputed": file_hashes == manifest["files_sha256"],
        "supervision_metadata_not_model_input": (
            manifest["relation_ids_are_supervision_and_audit_only"] is True),
        "no_features_or_scores_or_sealed_benchmarks_loaded": all((
            manifest["model_outputs_loaded"] is False,
            manifest["features_extracted"] is False,
            manifest["internal_test_scored"] is False,
            manifest["external_test_scored"] is False,
            manifest["sealed_benchmarks_loaded"] is False,
        )),
    }
    return {
        "protocol": AUDIT_PROTOCOL, "audit_passed": all(checks.values()),
        "checks": checks, "counts": counts,
        "manifest_sha256": sha(manifest_path), "files_sha256": file_hashes,
        "internal_test_scored": False, "external_test_scored": False,
        "sealed_benchmarks_loaded": False,
        "terminal_status": "completed_fewrel_selective_forgetting_pair_audit",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(args.data_root, args.source_root)
    write_json(Path(args.output), result)
    print(json.dumps(result, indent=2))
    if not result["audit_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
