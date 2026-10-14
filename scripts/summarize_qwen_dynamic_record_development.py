"""Merge dynamic-record training shards and apply the frozen development gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_qwen_cohort_forgetting import comparison
from tasks.train_qwen_dynamic_record_forgetting import POLICIES, PROTOCOL, SEEDS


def multi_seed_summary(per_seed):
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
            "constraint_violations": sum(row["constraint_violations"] for row in rows),
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


def run(args):
    import torch

    root, output = Path(args.root), Path(args.output)
    shards = [root / name for name in args.shards.split(",") if name]
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    per_seed, controllers, shard_meta, seen = {}, {}, [], set()
    checkpoint_metadata = None
    data_hash = resource_hash = None
    for shard in shards:
        manifest = json.loads((shard / "manifest.json").read_text())
        training = json.loads((shard / "training.json").read_text())
        results = json.loads((shard / "seed_results.json").read_text())
        checkpoint = torch.load(
            shard / "controllers.pt", map_location="cpu", weights_only=True
        )
        if manifest.get("protocol") != PROTOCOL or checkpoint.get("protocol") != PROTOCOL:
            raise ValueError("unexpected dynamic development shard protocol")
        metadata = {
            key: checkpoint[key]
            for key in ("controller_seeds", "features", "actions")
        }
        if checkpoint_metadata is None:
            checkpoint_metadata = metadata
        elif metadata != checkpoint_metadata:
            raise ValueError("development shards used different controller metadata")
        data_hash = data_hash or manifest["data_manifest_sha256"]
        resource_hash = resource_hash or manifest["resource_audit_sha256"]
        if manifest["data_manifest_sha256"] != data_hash:
            raise ValueError("development shards used different corpora")
        if manifest["resource_audit_sha256"] != resource_hash:
            raise ValueError("development shards used different resource releases")
        for seed in manifest["projection_seeds"]:
            key = str(seed)
            if seed in seen or key not in results or key not in checkpoint["controllers"]:
                raise ValueError("duplicate or incomplete projection shard")
            seen.add(seed); per_seed[key] = results[key]
            controllers[key] = checkpoint["controllers"][key]
        for name, digest in training["raw_rows_sha256"].items():
            if sha(shard / name) != digest:
                raise ValueError("development raw row hash mismatch")
        if sha(shard / "controllers.pt") != training["checkpoint_sha256"]:
            raise ValueError("development controller hash mismatch")
        shard_meta.append({
            "path": shard.name,
            "manifest_sha256": sha(shard / "manifest.json"),
            "training_sha256": sha(shard / "training.json"),
            "seed_results_sha256": sha(shard / "seed_results.json"),
            "checkpoint_sha256": sha(shard / "controllers.pt"),
            "seconds": training["seconds"],
            "overall_peak_allocated_gib": training["overall_peak_allocated_gib"],
            "overall_peak_reserved_gib": training["overall_peak_reserved_gib"],
        })
    if seen != set(SEEDS):
        raise ValueError("development merge requires all three projections")
    summary = multi_seed_summary(per_seed)
    gate = development_gate(summary)
    checkpoint_path = output / "controllers.pt"
    torch.save({
        "protocol": PROTOCOL,
        "projection_seeds": list(SEEDS),
        **checkpoint_metadata,
        "controllers": controllers,
    }, checkpoint_path)
    result = {
        "protocol": PROTOCOL,
        "development": summary,
        "development_gate": gate,
        "stage_passed": gate["passed"],
        "test_generated": False,
        "test_scored": False,
        "next_stage": (
            "independent_development_audit_before_test"
            if gate["passed"] else "stop_dynamic_record_branch"
        ),
        "terminal_status": "completed_qwen_dynamic_record_development",
    }
    training = {
        "two_gpu_critical_path_seconds": max(row["seconds"] for row in shard_meta),
        "overall_peak_allocated_gib": max(
            row["overall_peak_allocated_gib"] for row in shard_meta
        ),
        "overall_peak_reserved_gib": max(
            row["overall_peak_reserved_gib"] for row in shard_meta
        ),
        "shards": shard_meta,
        "checkpoint_sha256": sha(checkpoint_path),
    }
    manifest = {
        "protocol": PROTOCOL,
        "data_manifest_sha256": data_hash,
        "resource_audit_sha256": resource_hash,
        "projection_seeds": list(SEEDS),
        "policies": list(POLICIES),
        "test_generated": False,
        "test_scored": False,
        "shards": [row["path"] for row in shard_meta],
    }
    write_json(output / "summary.json", result)
    write_json(output / "training.json", training)
    write_json(output / "manifest.json", manifest)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shards", default="shard_gpu0,shard_gpu1")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
