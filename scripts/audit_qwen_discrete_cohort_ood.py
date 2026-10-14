"""Independently audit the frozen-controller lifecycle stress evaluation."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics

from tasks.build_qwen_discrete_cohort_ood import (
    BLOCKS,
    BOUNDARIES,
    PROTOCOL as DATA_PROTOCOL,
    SUITES,
    load_corpus,
)
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_associative_binding import _bootstrap
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "audit_qwen_discrete_cohort_ood_v1"
RESULT_PROTOCOL = "qwen_discrete_cohort_ood_v1"
CONFIRMATION_PROTOCOL = "audit_qwen_discrete_cohort_confirmation_v1"
SEEDS = (2301, 2302, 2303)
POLICIES = (
    "fixed_retain",
    "fixed_global_clear",
    "fixed_global_half",
    "fixed_window2",
    "fixed_window3",
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
    queries = [query for episode in records for query in episode["queries"]]
    decisions = [decision for episode in records for decision in episode["decisions"]]
    groups, families, roles = {}, {}, {}
    for group in sorted({episode["group_id"] for episode in records}):
        values = [
            query
            for episode in records
            if episode["group_id"] == group
            for query in episode["queries"]
        ]
        groups[group] = {
            "accuracy": statistics.fmean(row["correct"] for row in values),
            "nll": statistics.fmean(row["nll"] for row in values),
        }
    for family in sorted({episode["family"] for episode in records}):
        values = [
            query
            for episode in records
            if episode["family"] == family
            for query in episode["queries"]
        ]
        families[family] = statistics.fmean(row["correct"] for row in values)
    for role in ("current", "historical", "anchor"):
        values = [query for query in queries if query["role"] == role]
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
    result = {}
    for policy in POLICIES:
        rows = [per_seed[str(seed)][policy] for seed in SEEDS]
        result[policy] = {
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
    return result


def comparison(primary, baseline, seed):
    groups = sorted(next(iter(primary["per_seed"].values()))["groups"])
    accuracy, nll = [], []
    for group in groups:
        accuracy.append(
            statistics.fmean(
                primary["per_seed"][str(projection)]["groups"][group]["accuracy"]
                - baseline["per_seed"][str(projection)]["groups"][group]["accuracy"]
                for projection in SEEDS
            )
        )
        nll.append(
            statistics.fmean(
                baseline["per_seed"][str(projection)]["groups"][group]["nll"]
                - primary["per_seed"][str(projection)]["groups"][group]["nll"]
                for projection in SEEDS
            )
        )
    return {
        "accuracy_gain": statistics.fmean(accuracy),
        "accuracy_bootstrap_95ci": _bootstrap(accuracy, seed),
        "nll_gain": statistics.fmean(nll),
        "nll_bootstrap_95ci": _bootstrap(nll, seed + 1),
    }


def action_summary(records_by_seed):
    totals = {
        "endpoint": Counter(decisions=0, matches=0, queries=0, correct=0),
        "interior": Counter(decisions=0, matches=0, queries=0, correct=0),
    }
    mae = {"endpoint": 0.0, "interior": 0.0}
    oracle_interior = predicted_interior = decisions = 0
    for records in records_by_seed.values():
        for episode in records:
            episode_decisions, queries = episode["decisions"], episode["queries"]
            if len(queries) != 3 * len(episode_decisions):
                raise ValueError("action audit requires three queries per decision")
            for index, decision in enumerate(episode_decisions):
                oracle = tuple(float(value) for value in decision["oracle_retention"])
                predicted = tuple(float(value) for value in decision["retention"])
                kind = "interior" if 0.5 in oracle else "endpoint"
                selected = queries[3 * index : 3 * index + 3]
                totals[kind]["decisions"] += 1
                totals[kind]["matches"] += int(predicted == oracle)
                totals[kind]["queries"] += 3
                totals[kind]["correct"] += sum(row["correct"] for row in selected)
                mae[kind] += float(decision["retention_mae"])
                oracle_interior += int(kind == "interior")
                predicted_interior += int(0.5 in predicted)
                decisions += 1
    if not decisions or not all(totals[kind]["decisions"] for kind in totals):
        raise ValueError("action audit requires endpoint and interior examples")
    return {
        "decisions": decisions,
        "oracle_interior_fraction": oracle_interior / decisions,
        "predicted_interior_fraction": predicted_interior / decisions,
        "buckets": {
            kind: {
                "decisions": values["decisions"],
                "action_accuracy": values["matches"] / values["decisions"],
                "query_accuracy": values["correct"] / values["queries"],
                "retention_mae": mae[kind] / values["decisions"],
            }
            for kind, values in totals.items()
        },
    }


def gate(suites, macro, action):
    primary = macro["grid_classifier"]
    physics = macro["physics_cohort"]
    oracle = macro["oracle_grid"]
    gain = comparison(primary, physics, 3011)
    checks = {
        "macro_accuracy_at_least_point_97": primary["mean_accuracy"] >= 0.97,
        "every_suite_accuracy_at_least_point_95": all(
            suites[suite]["grid_classifier"]["mean_accuracy"] >= 0.95
            for suite in SUITES
        ),
        "minimum_projection_seed_at_least_point_95": (
            primary["min_seed_accuracy"] >= 0.95
        ),
        "physics_gain_at_least_point_03": gain["accuracy_gain"] >= 0.03,
        "physics_gain_ci_lower_above_point_01": (
            gain["accuracy_bootstrap_95ci"][0] > 0.01
        ),
        "nll_gain_ci_lower_above_zero": gain["nll_bootstrap_95ci"][0] > 0.0,
        "no_suite_negative_physics_gain": all(
            suites[suite]["grid_classifier"]["mean_accuracy"]
            >= suites[suite]["physics_cohort"]["mean_accuracy"]
            for suite in SUITES
        ),
        "oracle_gap_at_most_point_02": (
            oracle["mean_accuracy"] - primary["mean_accuracy"] <= 0.02
        ),
        "current_and_anchor_at_least_point_98": min(
            primary["roles"]["current"], primary["roles"]["anchor"]
        )
        >= 0.98,
        "historical_at_least_point_93": primary["roles"]["historical"] >= 0.93,
        "endpoint_action_at_least_point_97": (
            action["buckets"]["endpoint"]["action_accuracy"] >= 0.97
        ),
        "interior_action_at_least_point_65": (
            action["buckets"]["interior"]["action_accuracy"] >= 0.65
        ),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "comparison_to_physics": gain,
        "oracle_gap": oracle["mean_accuracy"] - primary["mean_accuracy"],
    }


def source_truth(pairs):
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
        for inp, target in pairs
    }


def audit_rows(result_root, suite_pairs, recorded_hashes):
    suite_summaries, macro_records, hashes = {}, {
        str(seed): {policy: [] for policy in POLICIES} for seed in SEEDS
    }, {}
    grid_records = {str(seed): [] for seed in SEEDS}
    structure = True
    for suite in SUITES:
        truth = source_truth(suite_pairs[suite])
        per_seed = {}
        for seed in SEEDS:
            per_seed[str(seed)] = {}
            for policy in POLICIES:
                path = result_root / f"{suite}_seed{seed}_{policy}.jsonl"
                rows = read_jsonl(path)
                hashes[path.name] = sha(path)
                identifiers = [row["episode_id"] for row in rows]
                structure &= all(
                    (
                        len(rows) == len(truth),
                        len(set(identifiers)) == len(rows),
                        set(identifiers) == set(truth),
                        all(row["policy"] == policy for row in rows),
                        all(row["reader"] == "component_top1" for row in rows),
                    )
                )
                for row in rows:
                    source = truth[row["episode_id"]]
                    structure &= all(
                        (
                            row["group_id"] == source["group_id"],
                            row["family"] == source["family"] == suite,
                            [decision["block"] for decision in row["decisions"]]
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
                                (decision["oracle_retention"] is not None)
                                == (policy in ORACLE_RECORDED)
                                for decision in row["decisions"]
                            ),
                            all(
                                decision["oracle_retention"] is None
                                or (
                                    len(decision["oracle_retention"]) == 3
                                    and all(
                                        point in (0.0, 0.5, 1.0)
                                        for point in decision["oracle_retention"]
                                    )
                                )
                                for decision in row["decisions"]
                            ),
                        )
                    )
                per_seed[str(seed)][policy] = aggregate(rows)
                macro_records[str(seed)][policy].extend(rows)
                if policy == "grid_classifier":
                    grid_records[str(seed)].extend(rows)
        suite_summaries[suite] = multi_seed(per_seed)
    macro = multi_seed(
        {
            str(seed): {
                policy: aggregate(macro_records[str(seed)][policy])
                for policy in POLICIES
            }
            for seed in SEEDS
        }
    )
    return (
        suite_summaries,
        macro,
        action_summary(grid_records),
        structure,
        hashes,
        hashes == recorded_hashes,
    )


def audit(args):
    root = Path(args.root)
    result_root = root / "result"
    summary_path = result_root / "summary.json"
    training_path = result_root / "training.json"
    manifest_path = result_root / "manifest.json"
    summary = json.loads(summary_path.read_text())
    training = json.loads(training_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    confirmation = json.loads(Path(args.confirmation_audit).read_text())
    data_manifest, suite_pairs = load_corpus(args.data_root)
    suites, macro, action, structure, raw_hashes, raw_hash_match = audit_rows(
        result_root, suite_pairs, training["raw_rows_sha256"]
    )
    reconstructed_gate = gate(suites, macro, action)
    confirmation_summary = (
        Path(args.confirmation_root) / "result" / "summary.json"
    )
    confirmation_checkpoint = (
        Path(args.confirmation_root) / "result" / "controllers.pt"
    )
    checks = {
        "runner_and_terminal_completed": (
            (root / "runner_exit_code.txt").read_text().strip() == "0"
            and (root / "terminal_status.txt").read_text().strip()
            == "completed_qwen_discrete_cohort_ood"
        ),
        "result_protocol_and_frozen_controller": all(
            (
                summary["protocol"] == RESULT_PROTOCOL,
                manifest["protocol"] == RESULT_PROTOCOL,
                manifest["controller_retrained"] is False,
                manifest["calibration_performed"] is False,
                tuple(manifest["projection_seeds"]) == SEEDS,
                tuple(manifest["policies"]) == POLICIES,
            )
        ),
        "confirmation_release_and_hash_binding": all(
            (
                confirmation["protocol"] == CONFIRMATION_PROTOCOL,
                confirmation["audit_passed"] is True,
                confirmation["test_scored"] is True,
                confirmation["test_gate"]["passed"] is True,
                manifest["confirmation_audit_sha256"]
                == sha(args.confirmation_audit),
                manifest["confirmation_summary_sha256"]
                == sha(confirmation_summary),
                manifest["confirmation_checkpoint_sha256"]
                == sha(confirmation_checkpoint),
                confirmation["summary_sha256"] == sha(confirmation_summary),
                confirmation["checkpoint_sha256"] == sha(confirmation_checkpoint),
            )
        ),
        "fresh_corpus_protocol_seed_counts_and_audit": all(
            (
                data_manifest["protocol"] == DATA_PROTOCOL,
                data_manifest["seed"] == 202610154,
                data_manifest["groups_per_suite"] == 64,
                tuple(data_manifest["suites"]) == SUITES,
                data_manifest["blocks"] == BLOCKS,
                data_manifest["boundaries"]
                == {name: list(values) for name, values in BOUNDARIES.items()},
                data_manifest["audit"]["causal_answers_verified"] is True,
                data_manifest["audit"]["suite_identifiers_and_values_disjoint"]
                is True,
                data_manifest["audit"]["complete_twins"] is True,
                data_manifest["audit"]["oracle_actions_generated"] is False,
                manifest["data_manifest_sha256"]
                == sha(Path(args.data_root) / "manifest.json"),
            )
        ),
        "frozen_geometry_hashes": (
            manifest["geometry_hashes"] == GEOMETRY_HASHES
            and all(
                sha(Path(args.geometry_root) / name) == digest
                for name, digest in GEOMETRY_HASHES.items()
            )
        ),
        "raw_rows_exact_source_view": structure,
        "raw_row_hashes": raw_hash_match,
        "all_recorded_raw_hashes_accounted": raw_hashes == training["raw_rows_sha256"],
        "suite_aggregates_recomputed": close(suites, summary["suites"]),
        "macro_aggregates_recomputed": close(macro, summary["macro"]),
        "action_diagnostic_recomputed": close(action, summary["action_diagnostic"]),
        "registered_gate_recomputed": close(reconstructed_gate, summary["gate"]),
        "terminal_stage_matches_gate": (
            summary["stage_passed"] == reconstructed_gate["passed"]
        ),
    }
    result = {
        "protocol": PROTOCOL,
        "audit_passed": all(checks.values()),
        "checks": checks,
        "summary_sha256": sha(summary_path),
        "training_sha256": sha(training_path),
        "manifest_sha256": sha(manifest_path),
        "data_manifest_sha256": sha(Path(args.data_root) / "manifest.json"),
        "suites": suites,
        "macro": macro,
        "action_diagnostic": action,
        "gate": reconstructed_gate,
        "stage_passed": reconstructed_gate["passed"],
        "terminal_status": "completed_qwen_discrete_cohort_ood_audit",
    }
    write_json(Path(args.output), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--confirmation-root", required=True)
    parser.add_argument("--confirmation-audit", required=True)
    parser.add_argument("--geometry-root", required=True)
    parser.add_argument("--output", required=True)
    result = audit(parser.parse_args())
    print(json.dumps(result, indent=2))
    if not result["audit_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
