"""Independently audit dynamic record-level forgetting development results."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics

from scripts.audit_qwen_sparse_cohort_forgetting import close, comparison
from tasks.build_qwen_dynamic_record_forgetting import (
    DATA_SEED,
    FAMILIES,
    PROTOCOL as DATA_PROTOCOL,
    load_split,
    validate,
)
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "qwen_dynamic_record_forgetting_development_v1"
SEEDS = (2301, 2302, 2303)
CONTROLLER_SEEDS = {2301: 5101, 2302: 5102, 2303: 5103}
POLICIES = (
    "fixed_retain",
    "deterministic_random",
    "old_cohort",
    "cohort_physics",
    "exact_reference",
    "dynamic_rules",
    "slot_mlp",
    "set_unconstrained",
    "primary",
    "oracle",
)
COMPARABLE = {
    "fixed_retain",
    "deterministic_random",
    "dynamic_rules",
    "slot_mlp",
    "set_unconstrained",
    "primary",
    "oracle",
}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def source_truth(pairs):
    output = {}
    for inp, target in pairs:
        answers = {
            row["id"]: (row["answer"], row["role"], row["key"])
            for row in target["answers"]
        }
        output[inp["id"]] = {
            "group_id": inp["group_id"],
            "family": target["family"],
            "boundaries": inp["decision_boundaries"],
            "answers": answers,
        }
    return output


def aggregate(records):
    queries = [row for episode in records for row in episode["queries"]]
    decisions = [row for episode in records for row in episode["decisions"]]
    groups, families, roles = {}, {}, {}
    for group in sorted({row["group_id"] for row in records}):
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
    for family in sorted({row["family"] for row in records}):
        values = [
            query
            for episode in records
            if episode["family"] == family
            for query in episode["queries"]
        ]
        families[family] = statistics.fmean(row["correct"] for row in values)
    for role in ("current", "historical", "anchor"):
        values = [row for row in queries if row["role"] == role]
        roles[role] = statistics.fmean(row["correct"] for row in values)
    comparable = [row for row in decisions if row["retention_mae"] is not None]
    output = {
        "accuracy": statistics.fmean(row["correct"] for row in queries),
        "nll": statistics.fmean(row["nll"] for row in queries),
        "roles": roles,
        "families": families,
        "groups": groups,
        "retention_mae": (
            statistics.fmean(row["retention_mae"] for row in comparable)
            if comparable else None
        ),
        "grid_action_accuracy": (
            statistics.fmean(row["grid_action_match"] for row in comparable)
            if comparable else None
        ),
        "mean_retained_mass": statistics.fmean(
            row["retained_mass"] for row in decisions
        ),
        "queries": len(queries),
        "decisions": len(decisions),
        "selection_recall": statistics.fmean(
            row["selection_recall"] for row in decisions
        ),
        "constraint_violations": sum(
            row["constraint_violation"] for row in decisions
        ),
    }
    for interior in (False, True):
        bucket = [
            row for row in decisions
            if row["oracle_interior"] is interior
            and row["grid_action_match"] is not None
        ]
        prefix = "interior" if interior else "endpoint"
        output[f"{prefix}_action_accuracy"] = (
            statistics.fmean(row["grid_action_match"] for row in bucket)
            if bucket else None
        )
        output[f"{prefix}_decisions"] = len(bucket)
    return output


def multi_seed(per_seed):
    output = {}
    for policy in POLICIES:
        rows = [per_seed[str(seed)][policy] for seed in SEEDS]
        output[policy] = {
            "mean_accuracy": statistics.fmean(row["accuracy"] for row in rows),
            "mean_nll": statistics.fmean(row["nll"] for row in rows),
            "min_seed_accuracy": min(row["accuracy"] for row in rows),
            "mean_retention_mae": (
                None if rows[0]["retention_mae"] is None else
                statistics.fmean(row["retention_mae"] for row in rows)
            ),
            "mean_grid_action_accuracy": (
                None if rows[0]["grid_action_accuracy"] is None else
                statistics.fmean(row["grid_action_accuracy"] for row in rows)
            ),
            "roles": {
                role: statistics.fmean(row["roles"][role] for row in rows)
                for role in rows[0]["roles"]
            },
            "families": {
                family: statistics.fmean(row["families"][family] for row in rows)
                for family in rows[0]["families"]
            },
            "selection_recall": statistics.fmean(
                row["selection_recall"] for row in rows
            ),
            "constraint_violations": sum(
                row["constraint_violations"] for row in rows
            ),
            "endpoint_action_accuracy": (
                statistics.fmean(row["endpoint_action_accuracy"] for row in rows)
                if rows[0]["endpoint_action_accuracy"] is not None else None
            ),
            "interior_action_accuracy": (
                statistics.fmean(row["interior_action_accuracy"] for row in rows)
                if rows[0]["interior_action_accuracy"] is not None else None
            ),
            "per_seed": {str(seed): per_seed[str(seed)][policy] for seed in SEEDS},
        }
    return output


def development_gate(summary):
    candidates = [name for name in POLICIES if name not in {"primary", "oracle"}]
    strongest = max(candidates, key=lambda name: summary[name]["mean_accuracy"])
    primary, oracle = summary["primary"], summary["oracle"]
    gain = comparison(primary, summary[strongest], 5109)
    checks = {
        "accuracy_at_least_point_95": primary["mean_accuracy"] >= 0.95,
        "every_family_at_least_point_90": min(primary["families"].values()) >= 0.90,
        "minimum_projection_seed_at_least_point_93": primary["min_seed_accuracy"] >= 0.93,
        "gain_over_strongest_at_least_point_03": gain["accuracy_gain"] >= 0.03,
        "gain_ci_lower_above_point_01": gain["accuracy_bootstrap_95ci"][0] > 0.01,
        "nll_gain_ci_lower_above_zero": gain["nll_bootstrap_95ci"][0] > 0.0,
        "oracle_gap_at_most_point_03": (
            oracle["mean_accuracy"] - primary["mean_accuracy"] <= 0.03
        ),
        "current_and_anchor_at_least_point_98": min(
            primary["roles"]["current"], primary["roles"]["anchor"]
        ) >= 0.98,
        "historical_at_least_point_90": primary["roles"]["historical"] >= 0.90,
        "candidate_selection_recall_at_least_point_98": (
            primary["selection_recall"] >= 0.98
        ),
        "endpoint_action_at_least_point_95": (
            primary["endpoint_action_accuracy"] is not None
            and primary["endpoint_action_accuracy"] >= 0.95
        ),
        "interior_action_at_least_point_65": (
            primary["interior_action_accuracy"] is not None
            and primary["interior_action_accuracy"] >= 0.65
        ),
        "zero_hard_constraint_violations": primary["constraint_violations"] == 0,
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "strongest_non_oracle_comparator": strongest,
        "comparison": gain,
        "oracle_gap": oracle["mean_accuracy"] - primary["mean_accuracy"],
    }


def audit_rows(root, truth):
    per_seed, hashes, structure = {}, {}, True
    seed_shards = {2301: "shard_gpu0", 2302: "shard_gpu1", 2303: "shard_gpu0"}
    for seed in SEEDS:
        per_seed[str(seed)] = {}
        shard = root / seed_shards[seed]
        training = json.loads((shard / "training.json").read_text())
        for policy in POLICIES:
            path = shard / f"dev_seed{seed}_{policy}.jsonl"
            rows = read_jsonl(path)
            hashes[f"{seed}/{path.name}"] = sha(path)
            structure &= all(
                (
                    len(rows) == len(truth),
                    len({row["episode_id"] for row in rows}) == len(rows),
                    set(row["episode_id"] for row in rows) == set(truth),
                    all(row["policy"] == policy for row in rows),
                    all(row["reader"] == "component_top1" for row in rows),
                    training["raw_rows_sha256"].get(path.name) == sha(path),
                )
            )
            for episode in rows:
                source = truth[episode["episode_id"]]
                structure &= all(
                    (
                        episode["group_id"] == source["group_id"],
                        episode["family"] == source["family"],
                        [row["block"] for row in episode["decisions"]]
                        == source["boundaries"],
                        len(episode["queries"]) == 3 * len(source["boundaries"]),
                        all(
                            (query["answer"], query["role"], query["key"])
                            == source["answers"][query["id"]]
                            for query in episode["queries"]
                        ),
                        all(
                            query["correct"] in (0, 1)
                            and math.isfinite(float(query["nll"]))
                            and query["components_used"] == 1
                            for query in episode["queries"]
                        ),
                        all(
                            row["selection_recall"] >= 0.0
                            and row["selection_recall"] <= 1.0
                            and row["constraint_violation"] in (0, 1)
                            and len(row["oracle_retention"]) == 4
                            and (row["retention"] is not None) == (policy in COMPARABLE)
                            for row in episode["decisions"]
                        ),
                    )
                )
            per_seed[str(seed)][policy] = aggregate(rows)
    return multi_seed(per_seed), structure, hashes


def run(args):
    import torch

    root, result = Path(args.root), Path(args.root) / "result"
    data = root / "data"
    summary = json.loads((result / "summary.json").read_text())
    training = json.loads((result / "training.json").read_text())
    manifest = json.loads((result / "manifest.json").read_text())
    data_manifest = json.loads((data / "manifest.json").read_text())
    resource = json.loads(Path(args.resource_audit).read_text())
    confirmation = json.loads(Path(args.confirmation_audit).read_text())
    train_pairs, dev_pairs = load_split(data, "train"), load_split(data, "dev")
    corpus_audit = validate(train_pairs + dev_pairs)
    reconstructed, row_structure, raw_hashes = audit_rows(
        root, source_truth(dev_pairs)
    )
    gate = development_gate(reconstructed)
    checkpoint_path = result / "controllers.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint_structure = all(
        (
            checkpoint.get("protocol") == PROTOCOL,
            tuple(checkpoint.get("projection_seeds", [])) == SEEDS,
            checkpoint.get("controller_seeds") == CONTROLLER_SEEDS,
            len(checkpoint.get("features", {}).get("global", ())) == 12,
            len(checkpoint.get("features", {}).get("slot", ())) == 14,
            len(checkpoint.get("actions", ())) == 54,
            set(checkpoint.get("controllers", {})) == {str(seed) for seed in SEEDS},
            all(
                set(checkpoint["controllers"][str(seed)])
                == {"primary", "set_unconstrained", "slot_mlp"}
                for seed in SEEDS
            ),
        )
    )
    forbidden = [
        path for path in root.rglob("*")
        if path.is_file()
        and ("test" in path.name.lower() or "ood" in path.name.lower())
        and path.name != "tests.log"
    ]
    checks = {
        "runner_and_terminal_completed": (
            (root / "runner_exit_code.txt").read_text().strip() == "0"
            and (root / "terminal_status.txt").read_text().strip()
            == "completed_qwen_dynamic_record_development"
        ),
        "development_corpus_protocol_seed_counts_and_audit": all(
            (
                data_manifest.get("protocol") == DATA_PROTOCOL,
                data_manifest.get("phase") == "development",
                data_manifest.get("seed") == DATA_SEED,
                data_manifest.get("group_counts") == {"train": 512, "dev": 64},
                corpus_audit == data_manifest.get("audit"),
                corpus_audit.get("causal_answers_verified") is True,
                corpus_audit.get("counterfactual_twins_verified") is True,
                corpus_audit.get("symbols_disjoint") is True,
                manifest.get("data_manifest_sha256") == sha(data / "manifest.json"),
            )
        ),
        "resource_release_passed_and_hashed": (
            resource.get("audit_passed") is True
            and resource.get("summary", {}).get("selected_formal_train_groups") == 512
            and manifest.get("resource_audit_sha256") == sha(args.resource_audit)
        ),
        "old_comparator_confirmation_passed_and_hashed": (
            confirmation.get("audit_passed") is True
            and confirmation.get("test_gate", {}).get("passed") is True
            and all(
                json.loads((root / shard / "manifest.json").read_text()).get(
                    "old_cohort_checkpoint_sha256"
                ) == confirmation.get("checkpoint_sha256")
                for shard in ("shard_gpu0", "shard_gpu1")
            )
        ),
        "three_projection_shards_complete": all(
            json.loads((root / shard / "progress.json").read_text()).get("stage")
            == "completed"
            for shard in ("shard_gpu0", "shard_gpu1")
        ),
        "development_rows_exact_source_view": row_structure,
        "all_thirty_raw_files_hashed": len(raw_hashes) == 30,
        "development_aggregates_recomputed": close(
            reconstructed, summary["development"]
        ),
        "development_gate_recomputed": gate == summary["development_gate"],
        "checkpoint_hash_and_structure": (
            training.get("checkpoint_sha256") == sha(checkpoint_path)
            and checkpoint_structure
        ),
        "resource_accounting_valid": (
            0.0 < training.get("two_gpu_critical_path_seconds", 0.0) < 6 * 3600
            and 0.0 < training.get("overall_peak_allocated_gib", 0.0) < 24.0
        ),
        "failed_gate_stops_branch": (
            gate["passed"] is False
            and summary.get("stage_passed") is False
            and summary.get("next_stage") == "stop_dynamic_record_branch"
        ),
        "sealed_splits_never_generated_or_scored": (
            data_manifest.get("sealed_test_generated") is False
            and data_manifest.get("external_ood_generated") is False
            and manifest.get("test_generated") is False
            and manifest.get("test_scored") is False
            and summary.get("test_generated") is False
            and summary.get("test_scored") is False
            and not forbidden
        ),
    }
    audit = {
        "protocol": "audit_qwen_dynamic_record_forgetting_development_v1",
        "audit_passed": all(checks.values()),
        "checks": checks,
        "summary_sha256": sha(result / "summary.json"),
        "training_sha256": sha(result / "training.json"),
        "manifest_sha256": sha(result / "manifest.json"),
        "checkpoint_sha256": sha(checkpoint_path),
        "raw_files": raw_hashes,
        "development": reconstructed,
        "development_gate": gate,
        "test_generated": False,
        "test_scored": False,
        "terminal_status": "completed_qwen_dynamic_record_development_audit",
    }
    write_json(args.output, audit)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--resource-audit", required=True)
    parser.add_argument("--confirmation-audit", required=True)
    parser.add_argument("--output", required=True)
    result = run(parser.parse_args())
    print(json.dumps(result, indent=2))
    if not result["audit_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
