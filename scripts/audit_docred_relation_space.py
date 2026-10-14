"""Independent structural audit for the DocRED relation-space corpus."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from tasks.build_docred_relation_space import (
    CAPS, DEV_MIN, FEWREL_MANIFEST_SHA256, HELDOUT_RETAINED_MIN, PARTITIONS,
    PROTOCOL, SEED, SOURCE_HASHES, TRAIN_MIN, counts, load_gzip_json,
    masked_evidence, relation_partitions,
)
from tasks.build_fewrel_selective_forgetting import (
    canonical_sha, load_jsonl, normalize, stable,
)
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


def audit(data_root, source_root, fewrel_root):
    data_root, source_root, fewrel_root = map(Path, (data_root, source_root, fewrel_root))
    manifest_path = data_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    instances = load_jsonl(data_root / "instances.jsonl")
    pairs = load_jsonl(data_root / "pairs.jsonl")
    exclusions = load_jsonl(data_root / "exclusions.jsonl")
    descriptions = load_jsonl(data_root / "train_relation_descriptions.jsonl")
    train = load_gzip_json(source_root / "train_annotated.json.gz")
    dev = load_gzip_json(source_root / "dev.json.gz")
    relation_info = load_gzip_json(source_root / "rel_info.json.gz")
    fewrel = json.loads((fewrel_root / "data/manifest.json").read_text())
    excluded = set(fewrel["partitions"]["internal_test"]
                   + fewrel["partitions"]["external_test"])
    partitions, train_counts, dev_counts = relation_partitions(
        train, dev, relation_info, excluded, manifest["seed"])

    source = {"train_annotated.json.gz": train, "dev.json.gz": dev}
    source_reproduction = True
    for row in instances:
        document = source[row["source_file"]][int(row["document_index"])]
        label = document["labels"][int(row["label_index"])]
        surface, sentences = masked_evidence(document, label)
        source_reproduction &= all((
            label["r"] == row["relation_id"],
            canonical_sha(document) == row["source_document_sha256"],
            canonical_sha(label) == row["source_label_sha256"],
            surface == row["surface"], sentences == row["selected_sentence_ids"],
            "relation_description" not in row,
        ))

    relation_sets = {partition: set(values) for partition, values in partitions.items()}
    relation_isolation = (
        manifest["partitions"] == partitions
        and sum(map(len, relation_sets.values()))
            == len(set().union(*relation_sets.values())) == 67
        and not (set().union(*relation_sets.values()) & excluded)
        and all(row["relation_id"] in relation_sets[row["partition"]]
                for row in instances))
    document_owners, surface_owners, surface_relations = (
        defaultdict(set), defaultdict(set), defaultdict(set))
    for row in instances:
        document_owners[(row["source_file"], row["document_index"])].add(row["partition"])
        surface_owners[normalize(row["surface"])].add(row["partition"])
        surface_relations[(row["partition"], normalize(row["surface"]))].add(
            row["relation_id"])
    isolation = (all(len(value) == 1 for value in document_owners.values())
                 and all(len(value) == 1 for value in surface_owners.values())
                 and all(len(value) == 1 for value in surface_relations.values()))

    instance_ids = {row["instance_id"]: row for row in instances}
    pair_semantics, pair_ids = True, True
    identities = set()
    for row in pairs:
        left, right = (instance_ids.get(row["left_instance_id"]),
                       instance_ids.get(row["right_instance_id"]))
        if left is None or right is None:
            pair_semantics = False; continue
        same = left["relation_id"] == right["relation_id"]
        pair_semantics &= all((
            left["partition"] == right["partition"] == row["partition"],
            row["left_relation_id"] == left["relation_id"],
            row["right_relation_id"] == right["relation_id"],
            row["left_document_id"] == left["document_id"],
            row["right_document_id"] == right["document_id"],
            int(row["same_relation"]) == int(same),
            int(row["locality_risk"]) == int(not same),
            (row["kind"] == "same_relation_positive") == same,
            row["kind"] != "same_relation_positive"
                or left["document_id"] != right["document_id"],
        ))
        pair_ids &= row["pair_id"] == stable(
            SEED, "docred-pair", row["kind"], left["instance_id"],
            right["instance_id"])[:24]
        identity = (row["partition"], row["kind"]) + tuple(sorted((
            left["instance_id"], right["instance_id"])))
        pair_semantics &= identity not in identities
        identities.add(identity)

    recomputed_counts = counts(instances, pairs, exclusions, partitions)
    relation_counts = Counter((row["partition"], row["relation_id"])
                              for row in instances)
    coverage = all(
        HELDOUT_RETAINED_MIN <= relation_counts[(partition, relation)] <= CAPS[partition]
        if partition != "representation_train"
        else 0 < relation_counts[(partition, relation)] <= CAPS[partition]
        for partition, relations in partitions.items() for relation in relations)
    balanced = all(value["positive"] == value["negative"]
                   for value in recomputed_counts["by_partition"].values())
    expected_descriptions = [{"relation_id": relation,
                              "description": relation_info[relation]}
                             for relation in partitions["representation_train"]]
    file_hashes = {name: sha(data_root / name) for name in (
        "instances.jsonl", "pairs.jsonl", "exclusions.jsonl",
        "train_relation_descriptions.jsonl")}
    eligible_counts = {relation: {"train": train_counts[relation],
                                  "dev": dev_counts[relation]}
                       for values in partitions.values() for relation in values}
    checks = {
        "protocol_and_frozen_parameters": (
            manifest["protocol"] == PROTOCOL and manifest["seed"] == SEED
            and manifest["caps"] == CAPS
            and manifest["eligibility"] == {
                "train_evidence_minimum": TRAIN_MIN,
                "dev_evidence_minimum": DEV_MIN,
                "heldout_retained_minimum": HELDOUT_RETAINED_MIN}),
        "official_source_hashes": (
            {name: sha(source_root / name) for name in SOURCE_HASHES}
            == SOURCE_HASHES == manifest["source_files_sha256"]),
        "frozen_fewrel_manifest_and_target_exclusion": (
            sha(fewrel_root / "data/manifest.json") == FEWREL_MANIFEST_SHA256
            and manifest["fewrel_manifest_sha256"] == FEWREL_MANIFEST_SHA256
            and manifest["excluded_fewrel_test_relation_ids"] == sorted(excluded)),
        "eligible_counts_and_partitions_recomputed": (
            manifest["partitions"] == partitions
            and manifest["eligible_source_counts"] == eligible_counts),
        "all_instances_reproduced_from_source": bool(source_reproduction),
        "relation_and_target_isolation": relation_isolation,
        "document_surface_and_ambiguity_isolation": isolation,
        "unique_instances": len(instance_ids) == len(instances),
        "pair_references_labels_kinds_and_documents": bool(pair_semantics),
        "pair_ids_reproduced": bool(pair_ids),
        "unique_pair_ids": len({row["pair_id"] for row in pairs}) == len(pairs),
        "retained_relation_coverage": coverage,
        "balanced_pairs": balanced,
        "counts_recomputed": recomputed_counts == manifest["counts"],
        "training_descriptions_only_and_exact": descriptions == expected_descriptions,
        "instance_rows_do_not_release_relation_descriptions": all(
            "relation_description" not in row for row in instances),
        "file_hashes_recomputed": file_hashes == manifest["files_sha256"],
        "no_features_scores_or_test_access": all((
            manifest["model_outputs_loaded"] is False,
            manifest["features_extracted"] is False,
            manifest["relation_source_test_scored"] is False,
            manifest["fewrel_internal_test_loaded"] is False,
            manifest["fewrel_external_test_loaded"] is False,
            manifest["sealed_benchmarks_loaded"] is False,
        )),
    }
    return {
        "protocol": "audit_docred_relation_description_space_v1",
        "audit_passed": all(checks.values()), "checks": checks,
        "counts": recomputed_counts, "manifest_sha256": sha(manifest_path),
        "files_sha256": file_hashes,
        "relation_source_test_scored": False,
        "fewrel_internal_test_loaded": False, "fewrel_external_test_loaded": False,
        "sealed_benchmarks_loaded": False,
        "terminal_status": "completed_docred_relation_space_data_audit",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--fewrel-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(args.data_root, args.source_root, args.fewrel_root)
    write_json(Path(args.output), result)
    print(json.dumps(result, indent=2))
    if not result["audit_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
