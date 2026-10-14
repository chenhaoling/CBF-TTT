"""Independent row-level audit for the Qwen sparse-reader diagnostic."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


SEEDS = (2301, 2302, 2303)
READERS = (
    "dense_all",
    "component_top1",
    "component_top2",
    "component_top4",
    "component_top8",
    "component_top16",
    "exact_key_upper",
)
ELIGIBLE = READERS[1:-1]
POLICIES = ("fixed_retain", "physics_cohort", "exact_reference", "oracle_grid")
GEOMETRY_HASHES = {
    "summary.json": "4fa6e0aba9fa5ab335ee9650b31547379a7525cd4c52db3b02d49aa49fbd272c",
    "training.json": "01ecc4e9e5875485750b1f5a6121826eb254d9d865442a2c45ba79456ebb170c",
    "training_manifest.json": "d13cfa35397f65ed5231aace10d0338480482f08aba3f14f4e4f5b55c9347510",
    "contextual_transforms.pt": "524d213a82f158fddadab8973a28770a687138f528e269659430490affe74ae5",
}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def close(left, right, tolerance=1e-9):
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            close(left[key], right[key], tolerance) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            close(a, b, tolerance) for a, b in zip(left, right)
        )
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=0, abs_tol=tolerance)
    return left == right


def aggregate(records):
    queries = [row for episode in records for row in episode["queries"]]
    decisions = [row for episode in records for row in episode["decisions"]]
    groups, families, roles = {}, {}, {}
    for group in sorted({row["group_id"] for row in records}):
        values = [
            q
            for episode in records
            if episode["group_id"] == group
            for q in episode["queries"]
        ]
        groups[group] = {
            "accuracy": statistics.fmean(q["correct"] for q in values),
            "nll": statistics.fmean(q["nll"] for q in values),
        }
    for family in sorted({row["family"] for row in records}):
        values = [
            q
            for episode in records
            if episode["family"] == family
            for q in episode["queries"]
        ]
        families[family] = statistics.fmean(q["correct"] for q in values)
    for role in ("current", "historical", "anchor"):
        values = [row for row in queries if row["role"] == role]
        roles[role] = statistics.fmean(row["correct"] for row in values)
    return {
        "accuracy": statistics.fmean(row["correct"] for row in queries),
        "nll": statistics.fmean(row["nll"] for row in queries),
        "roles": roles,
        "families": families,
        "groups": groups,
        "mean_components_used": statistics.fmean(
            row["components_used"] for row in queries
        ),
        "mean_retained_mass": statistics.fmean(
            row["retained_mass"] for row in decisions
        ),
        "queries": len(queries),
        "decisions": len(decisions),
    }


def multi_seed(per_seed):
    output = {}
    for reader in READERS:
        output[reader] = {}
        for policy in POLICIES:
            rows = [per_seed[str(seed)][reader][policy] for seed in SEEDS]
            output[reader][policy] = {
                "mean_accuracy": statistics.fmean(row["accuracy"] for row in rows),
                "mean_nll": statistics.fmean(row["nll"] for row in rows),
                "min_seed_accuracy": min(row["accuracy"] for row in rows),
                "mean_components_used": statistics.fmean(
                    row["mean_components_used"] for row in rows
                ),
                "mean_retained_mass": statistics.fmean(
                    row["mean_retained_mass"] for row in rows
                ),
                "roles": {
                    role: statistics.fmean(row["roles"][role] for row in rows)
                    for role in rows[0]["roles"]
                },
                "families": {
                    family: statistics.fmean(
                        row["families"][family] for row in rows
                    )
                    for family in rows[0]["families"]
                },
                "per_seed": {
                    str(seed): per_seed[str(seed)][reader][policy] for seed in SEEDS
                },
            }
    return output


def decision(summary):
    dense = summary["dense_all"]["oracle_grid"]
    checks, passing = {}, []
    for reader in ELIGIBLE:
        row = summary[reader]["oracle_grid"]
        values = {
            "accuracy_at_least_point_95": row["mean_accuracy"] >= 0.95,
            "current_at_least_point_95": row["roles"]["current"] >= 0.95,
            "anchor_at_least_point_95": row["roles"]["anchor"] >= 0.95,
            "historical_at_least_point_90": row["roles"]["historical"] >= 0.90,
            "min_family_at_least_point_85": min(row["families"].values()) >= 0.85,
            "every_seed_improves_dense": all(
                row["per_seed"][str(seed)]["accuracy"]
                > dense["per_seed"][str(seed)]["accuracy"]
                for seed in SEEDS
            ),
        }
        checks[reader] = values
        if all(values.values()):
            passing.append(reader)
    if passing:
        best = max(summary[name]["oracle_grid"]["mean_accuracy"] for name in passing)
        selected = next(
            name
            for name in passing
            if best - summary[name]["oracle_grid"]["mean_accuracy"] < 0.005
        )
    else:
        selected = None
    upper = summary["exact_key_upper"]["oracle_grid"]
    upper_checks = {
        "accuracy_at_least_point_95": upper["mean_accuracy"] >= 0.95,
        "current_at_least_point_95": upper["roles"]["current"] >= 0.95,
        "anchor_at_least_point_95": upper["roles"]["anchor"] >= 0.95,
        "historical_at_least_point_90": upper["roles"]["historical"] >= 0.90,
        "min_family_at_least_point_85": min(upper["families"].values()) >= 0.85,
    }
    return {
        "checks": checks,
        "exact_key_upper_checks": upper_checks,
        "passed": selected is not None,
        "selected": selected,
    }


def source_truth(data_root):
    inputs = read_jsonl(Path(data_root) / "dev.inputs.jsonl")
    targets = read_jsonl(Path(data_root) / "dev.targets.jsonl")
    return {
        inp["id"]: {
            "group_id": inp["group_id"],
            "family": target["family"],
            "boundaries": inp["decision_boundaries"],
            "answers": {
                row["id"]: (row["answer"], row["role"], row["key"])
                for row in target["answers"]
            },
        }
        for inp, target in zip(inputs, targets)
    }


def audit_rows(result_root, data_root, recorded_hashes):
    truth = source_truth(data_root)
    per_seed, structure, observed_hashes = {}, True, {}
    for seed in SEEDS:
        per_seed[str(seed)] = {}
        for reader in READERS:
            per_seed[str(seed)][reader] = {}
            for policy in POLICIES:
                path = (
                    Path(result_root)
                    / f"dev_seed{seed}_{reader}_{policy}.jsonl"
                )
                rows = read_jsonl(path)
                observed_hashes[path.name] = sha(path)
                structure &= all(
                    (
                        len(rows) == len(truth),
                        len({row["episode_id"] for row in rows}) == len(rows),
                        set(row["episode_id"] for row in rows) == set(truth),
                        all(row["reader"] == reader for row in rows),
                        all(row["policy"] == policy for row in rows),
                    )
                )
                for row in rows:
                    source = truth[row["episode_id"]]
                    structure &= all(
                        (
                            row["group_id"] == source["group_id"],
                            row["family"] == source["family"],
                            [value["block"] for value in row["decisions"]]
                            == source["boundaries"],
                            len(row["queries"]) == 3 * len(source["boundaries"]),
                            all(
                                (query["answer"], query["role"], query["key"])
                                == source["answers"][query["id"]]
                                for query in row["queries"]
                            ),
                            all(
                                query["correct"] in (0, 1)
                                and math.isfinite(float(query["nll"]))
                                and query["components_used"] >= 0
                                for query in row["queries"]
                            ),
                            all(
                                value["retention"] is None
                                or (
                                    len(value["retention"]) == 3
                                    and all(0 <= point <= 1 for point in value["retention"])
                                )
                                for value in row["decisions"]
                            ),
                        )
                    )
                per_seed[str(seed)][reader][policy] = aggregate(rows)
    return (
        multi_seed(per_seed),
        structure,
        observed_hashes,
        observed_hashes == recorded_hashes,
    )


def audit(args):
    root, data_root = Path(args.root), Path(args.data_root)
    result_root = root / "result"
    summary_path = result_root / "summary.json"
    training_path = result_root / "training.json"
    manifest_path = result_root / "manifest.json"
    summary = json.loads(summary_path.read_text())
    training = json.loads(training_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    data_manifest = json.loads((data_root / "manifest.json").read_text())
    dev, row_structure, row_hashes, row_hash_match = audit_rows(
        result_root, data_root, training["raw_rows_sha256"]
    )
    reconstructed = decision(dev)
    raw_files = list(result_root.glob("dev_seed*_*.jsonl"))
    test_files = list(result_root.glob("test*"))
    checks = {
        "runner_and_terminal_completed": (
            (root / "runner_exit_code.txt").read_text().strip() == "0"
            and (root / "terminal_status.txt").read_text().strip()
            == "completed_qwen_sparse_reader_diagnostic"
        ),
        "fresh_corpus_source_locked": all(
            (
                data_manifest["protocol"] == "cbf_event_curriculum_v1",
                data_manifest["seed"] == 202610151,
                data_manifest["group_counts"]
                == {"train": 128, "dev": 32, "test": 64},
                data_manifest["audit"]["causal_answers_verified"] is True,
                data_manifest["audit"]["entities_disjoint"] is True,
                data_manifest["audit"]["templates_disjoint"] is True,
                manifest["data_manifest_sha256"] == sha(data_root / "manifest.json"),
            )
        ),
        "frozen_geometry_hashes": manifest["geometry_hashes"] == GEOMETRY_HASHES,
        "registered_configuration": all(
            (
                manifest["protocol"] == "qwen_sparse_reader_diagnostic_v1",
                tuple(manifest["readers"]) == READERS,
                tuple(manifest["eligible_readers"]) == ELIGIBLE,
                tuple(manifest["policies"]) == POLICIES,
                tuple(manifest["projection_seeds"]) == SEEDS,
                manifest["splits_loaded"] == ["train", "dev"],
                manifest["source_group_count"] == 32,
            )
        ),
        "development_rows_exact_source_view": row_structure,
        "development_row_hashes": row_hash_match,
        "development_aggregates_recomputed": close(dev, summary["dev"]),
        "development_decision_recomputed": reconstructed == summary["decision"],
        "all_recorded_rows_accounted": all(
            (
                len(raw_files) == len(SEEDS) * len(READERS) * len(POLICIES),
                row_hashes == training["raw_rows_sha256"],
            )
        ),
        "test_remained_sealed": all(
            (
                not test_files,
                manifest["test_loaded"] is False,
                manifest["test_scored"] is False,
                summary["test_scored"] is False,
            )
        ),
        "terminal_stage_matches_decision": (
            summary["stage_passed"] == reconstructed["passed"]
            and (
                summary["next_stage"] == "freeze_fresh_sparse_reader_experiment"
            )
            == reconstructed["passed"]
        ),
    }
    result = {
        "protocol": "audit_qwen_sparse_reader_diagnostic_v1",
        "audit_passed": all(checks.values()),
        "checks": checks,
        "summary_sha256": sha(summary_path),
        "training_sha256": sha(training_path),
        "manifest_sha256": sha(manifest_path),
        "decision": reconstructed,
        "development": dev,
        "test_scored": False,
        "terminal_status": "completed_qwen_sparse_reader_diagnostic_audit",
    }
    write_json(Path(args.output), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    result = audit(parser.parse_args())
    print(json.dumps(result, indent=2))
    if not result["audit_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
