"""Audit the released natural semantic relation-pair corpus."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from tasks.build_natural_semantic_relation_pairs import (
    HARD_CANDIDATES,
    PROTOCOL,
    construct,
    counts,
    load_jsonl,
    normalize,
    verify_release,
)
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


def audit(data_root, inventory_root, headroom_root, output):
    data_root, output = Path(data_root), Path(output)
    rows_path, dependencies = verify_release(inventory_root, headroom_root)
    manifest = json.loads((data_root / "manifest.json").read_text())
    pairs = load_jsonl(data_root / "pairs.jsonl")
    surfaces = load_jsonl(data_root / "surfaces.jsonl")
    exclusions = load_jsonl(data_root / "excluded.jsonl")
    expected = construct(load_jsonl(rows_path), manifest["seed"],
                         manifest["relation_cap"])
    surface_index = {row["surface_id"]: row["text"] for row in surfaces}
    pair_structure = all((
        row["same_relation"] in (0, 1),
        row["locality_risk"] == 1 - row["same_relation"],
        row["left_surface_id"] in surface_index,
        row["right_surface_id"] in surface_index,
        surface_index[row["left_surface_id"]] == row["left_surface"],
        surface_index[row["right_surface_id"]] == row["right_surface"],
        (row["anchor_relation_id"] == row["other_relation_id"])
            == bool(row["same_relation"]),
        normalize(row["left_surface"]) != normalize(row["right_surface"]),
        row["kind"] in {"within_row_positive", "cross_row_positive",
                        "lexical_hard_negative", "random_negative"},
    ) for row in pairs)
    relation_folds = {}
    relation_disjoint = True
    for row in pairs:
        for relation in (row["anchor_relation_id"], row["other_relation_id"]):
            if relation in relation_folds and relation_folds[relation] != row["outer_fold"]:
                relation_disjoint = False
            relation_folds[relation] = row["outer_fold"]
    summary = counts(pairs, surfaces, exclusions)
    coverage = all(value["positive"] >= 500 and value["negative"] >= 500
                   and value["locality_risk_positive"] >= 500
                   for value in summary["by_fold"].values())
    checks = {
        "audited_release_dependencies": manifest.get("dependencies") == dependencies,
        "protocol_and_access_locks": all((
            manifest.get("protocol") == PROTOCOL,
            manifest.get("hard_candidate_pool") == HARD_CANDIDATES,
            manifest.get("relation_ids_are_training_labels_only") is True,
            manifest.get("model_outcomes_loaded") is False,
            manifest.get("embeddings_extracted") is False,
            manifest.get("sealed_benchmarks_loaded") is False,
            manifest.get("terminal_status")
                == "frozen_natural_semantic_pairs_before_embeddings",
        )),
        "deterministic_exact_reconstruction": (
            pairs == expected[0] and surfaces == expected[1] and exclusions == expected[2]),
        "file_hashes_exact": all(
            manifest["files_sha256"][name] == sha(data_root / name)
            for name in ("pairs.jsonl", "surfaces.jsonl", "excluded.jsonl")),
        "manifest_counts_exact": manifest.get("counts") == summary,
        "unique_pair_and_surface_identities": (
            len({row["pair_id"] for row in pairs}) == len(pairs)
            and len(surface_index) == len(surfaces)),
        "pair_labels_surfaces_and_relation_semantics": pair_structure,
        "relation_fold_isolation": relation_disjoint,
        "positive_negative_and_locality_risk_coverage": coverage,
    }
    result = {
        "protocol": "audit_natural_semantic_relation_pairs_v1",
        "checks": checks,
        "audit_passed": all(checks.values()),
        "manifest_sha256": sha(data_root / "manifest.json"),
        "files_sha256": {name: sha(data_root / name) for name in (
            "pairs.jsonl", "surfaces.jsonl", "excluded.jsonl")},
        "counts": summary,
    }
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--inventory-root", required=True)
    parser.add_argument("--headroom-root", required=True)
    parser.add_argument("--output", required=True)
    result = audit(**vars(parser.parse_args()))
    print(json.dumps(result, indent=2))
    if not result["audit_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
