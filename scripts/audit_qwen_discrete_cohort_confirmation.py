"""Independent audit for fresh discrete cohort-state confirmation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from scripts.audit_qwen_sparse_cohort_forgetting import (
    close,
    comparison,
    read_jsonl,
    source_truth,
)


SEEDS = (2301, 2302, 2303)
CONTROLLER_SEEDS = (2611, 2617, 2621)
FIXED = (
    "fixed_retain",
    "fixed_global_clear",
    "fixed_global_half",
    "fixed_window2",
    "fixed_window3",
)
POLICIES = FIXED + (
    "deterministic_random",
    "reference_blind",
    "learned_snapped",
    "grid_classifier",
    "learned",
    "physics_cohort",
    "exact_reference",
    "oracle_grid",
)
ORACLE_RECORDED = {
    "learned_snapped",
    "grid_classifier",
    "learned",
    "oracle_grid",
}
FEATURE_COUNT = 20
GEOMETRY_HASHES = {
    "summary.json": "4fa6e0aba9fa5ab335ee9650b31547379a7525cd4c52db3b02d49aa49fbd272c",
    "training.json": "01ecc4e9e5875485750b1f5a6121826eb254d9d865442a2c45ba79456ebb170c",
    "training_manifest.json": "d13cfa35397f65ed5231aace10d0338480482f08aba3f14f4e4f5b55c9347510",
    "contextual_transforms.pt": "524d213a82f158fddadab8973a28770a687138f528e269659430490affe74ae5",
}


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
    retained = [row for row in decisions if row["retention_mae"] is not None]
    return {
        "accuracy": statistics.fmean(row["correct"] for row in queries),
        "nll": statistics.fmean(row["nll"] for row in queries),
        "roles": roles,
        "families": families,
        "groups": groups,
        "retention_mae": (
            statistics.fmean(row["retention_mae"] for row in retained)
            if retained
            else None
        ),
        "grid_action_accuracy": (
            statistics.fmean(row["grid_action_match"] for row in retained)
            if retained
            else None
        ),
        "mean_retained_mass": statistics.fmean(
            row["retained_mass"] for row in decisions
        ),
        "queries": len(queries),
        "decisions": len(decisions),
    }


def multi_seed(per_seed):
    output = {}
    for policy in POLICIES:
        rows = [per_seed[str(seed)][policy] for seed in SEEDS]
        output[policy] = {
            "mean_accuracy": statistics.fmean(row["accuracy"] for row in rows),
            "mean_nll": statistics.fmean(row["nll"] for row in rows),
            "min_seed_accuracy": min(row["accuracy"] for row in rows),
            "mean_retention_mae": (
                None
                if rows[0]["retention_mae"] is None
                else statistics.fmean(row["retention_mae"] for row in rows)
            ),
            "mean_grid_action_accuracy": (
                None
                if rows[0]["grid_action_accuracy"] is None
                else statistics.fmean(row["grid_action_accuracy"] for row in rows)
            ),
            "mean_retained_mass": statistics.fmean(
                row["mean_retained_mass"] for row in rows
            ),
            "roles": {
                role: statistics.fmean(row["roles"][role] for row in rows)
                for role in rows[0]["roles"]
            },
            "families": {
                family: statistics.fmean(row["families"][family] for row in rows)
                for family in rows[0]["families"]
            },
            "per_seed": {
                str(seed): per_seed[str(seed)][policy] for seed in SEEDS
            },
        }
    return output


def prerequisites(predecessor):
    dev = predecessor["development"]
    primary, physics, oracle = (
        dev["grid_classifier"],
        dev["physics_cohort"],
        dev["oracle_grid"],
    )
    return {
        "grid_accuracy_at_least_point_985": primary["mean_accuracy"] >= 0.985,
        "every_seed_at_least_point_98": min(
            row["accuracy"] for row in primary["per_seed"].values()
        )
        >= 0.98,
        "gain_over_physics_at_least_point_05": (
            primary["mean_accuracy"] - physics["mean_accuracy"] >= 0.05
        ),
        "oracle_gap_at_most_point_01": (
            oracle["mean_accuracy"] - primary["mean_accuracy"] <= 0.01
        ),
    }


def development_gate(summary, training_meta):
    primary, physics, oracle, blind = (
        summary["grid_classifier"],
        summary["physics_cohort"],
        summary["oracle_grid"],
        summary["reference_blind"],
    )
    gain = comparison(primary, physics, 2811)
    checks = {
        "accuracy_at_least_point_985": primary["mean_accuracy"] >= 0.985,
        "accuracy_gain_over_physics_at_least_point_05": gain["accuracy_gain"] >= 0.05,
        "accuracy_ci_lower_above_point_035": gain["accuracy_bootstrap_95ci"][0] > 0.035,
        "nll_ci_lower_above_zero": gain["nll_bootstrap_95ci"][0] > 0.0,
        "every_seed_gain_at_least_point_04": all(
            primary["per_seed"][str(seed)]["accuracy"]
            - physics["per_seed"][str(seed)]["accuracy"]
            >= 0.04
            for seed in SEEDS
        ),
        "oracle_gap_at_most_point_01": (
            oracle["mean_accuracy"] - primary["mean_accuracy"] <= 0.01
        ),
        "current_and_anchor_at_least_point_99": min(
            primary["roles"]["current"], primary["roles"]["anchor"]
        )
        >= 0.99,
        "historical_at_least_point_96": primary["roles"]["historical"] >= 0.96,
        "all_families_at_least_point_95": min(primary["families"].values()) >= 0.95,
        "grid_action_agreement_at_least_point_88": (
            primary["mean_grid_action_accuracy"] >= 0.88
        ),
        "gain_over_reference_blind_at_least_point_20": (
            primary["mean_accuracy"] - blind["mean_accuracy"] >= 0.20
        ),
        "interior_label_fraction_at_least_point_10": min(
            value["interior_fraction"] for value in training_meta.values()
        )
        >= 0.10,
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "comparison_to_physics": gain,
        "oracle_gap": oracle["mean_accuracy"] - primary["mean_accuracy"],
    }


def test_gate(summary):
    primary, physics, oracle = (
        summary["grid_classifier"],
        summary["physics_cohort"],
        summary["oracle_grid"],
    )
    gain = comparison(primary, physics, 2831)
    checks = {
        "accuracy_at_least_point_985": primary["mean_accuracy"] >= 0.985,
        "accuracy_gain_over_physics_at_least_point_05": gain["accuracy_gain"] >= 0.05,
        "accuracy_ci_lower_above_point_035": gain["accuracy_bootstrap_95ci"][0] > 0.035,
        "nll_ci_lower_above_zero": gain["nll_bootstrap_95ci"][0] > 0.0,
        "every_seed_gain_at_least_point_04": all(
            primary["per_seed"][str(seed)]["accuracy"]
            - physics["per_seed"][str(seed)]["accuracy"]
            >= 0.04
            for seed in SEEDS
        ),
        "oracle_gap_at_most_point_01": (
            oracle["mean_accuracy"] - primary["mean_accuracy"] <= 0.01
        ),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "comparison_to_physics": gain,
        "oracle_gap": oracle["mean_accuracy"] - primary["mean_accuracy"],
    }


def audit_rows(result_root, data_root, split, recorded_hashes):
    truth = source_truth(data_root, split)
    per_seed, structure, observed_hashes = {}, True, {}
    for seed in SEEDS:
        per_seed[str(seed)] = {}
        for policy in POLICIES:
            path = Path(result_root) / f"{split}_seed{seed}_{policy}.jsonl"
            rows = read_jsonl(path)
            observed_hashes[path.name] = sha(path)
            structure &= all(
                (
                    len(rows) == len(truth),
                    len({row["episode_id"] for row in rows}) == len(rows),
                    set(row["episode_id"] for row in rows) == set(truth),
                    all(row["policy"] == policy for row in rows),
                    all(row["reader"] == "component_top1" for row in rows),
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
                            and query["components_used"] == 1
                            for query in row["queries"]
                        ),
                        all(
                            (value["oracle_retention"] is not None)
                            == (policy in ORACLE_RECORDED)
                            for value in row["decisions"]
                        ),
                    )
                )
            per_seed[str(seed)][policy] = aggregate(rows)
    return (
        multi_seed(per_seed),
        structure,
        observed_hashes,
        observed_hashes
        == {name: recorded_hashes[name] for name in observed_hashes},
    )


def audit(args):
    import torch

    root, data_root = Path(args.root), Path(args.data_root)
    result_root = root / "result"
    summary_path = result_root / "summary.json"
    training_path = result_root / "training.json"
    manifest_path = result_root / "manifest.json"
    summary = json.loads(summary_path.read_text())
    training = json.loads(training_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    data_manifest = json.loads((data_root / "manifest.json").read_text())
    reader_audit = json.loads(Path(args.reader_audit).read_text())
    predecessor = json.loads(Path(args.predecessor_audit).read_text())
    reconstructed_prerequisites = prerequisites(predecessor)

    dev, dev_structure, dev_hashes, dev_hash_match = audit_rows(
        result_root, data_root, "dev", training["raw_rows_sha256"]
    )
    reconstructed_dev_gate = development_gate(dev, training["seeds"])
    test_expected = reconstructed_dev_gate["passed"]
    if test_expected:
        test, test_structure, test_hashes, test_hash_match = audit_rows(
            result_root, data_root, "test", training["raw_rows_sha256"]
        )
        reconstructed_test_gate = test_gate(test)
    else:
        test, test_structure, test_hash_match = None, True, True
        test_hashes, reconstructed_test_gate = {}, None

    checkpoint_path = result_root / "controllers.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint_structure = all(
        (
            checkpoint["protocol"] == "qwen_discrete_cohort_confirmation_v1",
            checkpoint["reader"] == "component_top1",
            len(checkpoint["features"]) == FEATURE_COUNT,
            len(checkpoint["grid"]) == 27,
            tuple(checkpoint["projection_seeds"]) == SEEDS,
            tuple(checkpoint["controller_seeds"]) == CONTROLLER_SEEDS,
            set(checkpoint["controllers"]) == {str(seed) for seed in SEEDS},
        )
    )
    test_files = list(result_root.glob("test_seed*_*.jsonl"))
    checks = {
        "runner_and_terminal_completed": (
            (root / "runner_exit_code.txt").read_text().strip() == "0"
            and (root / "terminal_status.txt").read_text().strip()
            == "completed_qwen_discrete_cohort_confirmation"
        ),
        "fresh_corpus_protocol_seed_counts_and_audit": all(
            (
                data_manifest["protocol"] == "cbf_event_curriculum_v1",
                data_manifest["seed"] == 202610153,
                data_manifest["group_counts"]
                == {"train": 256, "dev": 64, "test": 128},
                data_manifest["audit"]["causal_answers_verified"] is True,
                data_manifest["audit"]["entities_disjoint"] is True,
                data_manifest["audit"]["templates_disjoint"] is True,
                manifest["data_manifest_sha256"] == sha(data_root / "manifest.json"),
            )
        ),
        "reader_dependency_passed_and_hashed": all(
            (
                reader_audit["audit_passed"] is True,
                reader_audit["decision"]["selected"] == "component_top1",
                manifest["reader_audit_sha256"] == sha(args.reader_audit),
            )
        ),
        "predecessor_dependency_and_prerequisites": all(
            (
                predecessor["audit_passed"] is True,
                predecessor["test_scored"] is False,
                all(reconstructed_prerequisites.values()),
                manifest["predecessor_audit_sha256"] == sha(args.predecessor_audit),
                manifest["predecessor_prerequisites"] == reconstructed_prerequisites,
            )
        ),
        "frozen_geometry_hashes": manifest["geometry_hashes"] == GEOMETRY_HASHES,
        "checkpoint_hash_and_structure": (
            training["checkpoint_sha256"] == sha(checkpoint_path)
            and checkpoint_structure
        ),
        "development_rows_exact_source_view": dev_structure,
        "development_row_hashes": dev_hash_match,
        "development_aggregates_recomputed": close(dev, summary["dev"]),
        "development_gate_recomputed": reconstructed_dev_gate == summary["dev_gate"],
        "test_release_matches_gate": all(
            (
                manifest["test_loaded"] == test_expected,
                manifest["test_scored"] == test_expected,
                summary["test_scored"] == test_expected,
                bool(test_files) == test_expected,
            )
        ),
        "test_rows_exact_source_view_if_released": test_structure,
        "test_row_hashes_if_released": test_hash_match,
        "test_aggregates_recomputed_if_released": close(test, summary["test"]),
        "test_gate_recomputed_if_released": reconstructed_test_gate == summary["test_gate"],
        "all_recorded_raw_hashes_accounted": (
            {**dev_hashes, **test_hashes} == training["raw_rows_sha256"]
        ),
        "terminal_stage_matches_test_gate": (
            summary["stage_passed"]
            == bool(reconstructed_test_gate and reconstructed_test_gate["passed"])
        ),
    }
    result = {
        "protocol": "audit_qwen_discrete_cohort_confirmation_v1",
        "audit_passed": all(checks.values()),
        "checks": checks,
        "summary_sha256": sha(summary_path),
        "training_sha256": sha(training_path),
        "manifest_sha256": sha(manifest_path),
        "checkpoint_sha256": sha(checkpoint_path),
        "predecessor_prerequisites": reconstructed_prerequisites,
        "development_gate": reconstructed_dev_gate,
        "test_gate": reconstructed_test_gate,
        "development": dev,
        "test": test,
        "test_scored": test_expected,
        "terminal_status": "completed_qwen_discrete_cohort_confirmation_audit",
    }
    write_json(Path(args.output), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--reader-audit", required=True)
    parser.add_argument("--predecessor-audit", required=True)
    parser.add_argument("--output", required=True)
    result = audit(parser.parse_args())
    print(json.dumps(result, indent=2))
    if not result["audit_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
