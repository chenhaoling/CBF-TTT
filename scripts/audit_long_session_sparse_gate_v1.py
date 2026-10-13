"""Independently recompute the sparse long-session development result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from tasks.build_long_session_sparse_corpus import LENGTHS, PROTOCOL as DATA_PROTOCOL
from tasks.pack_cbf_event_warmup import sha
from tasks.train_long_session_sparse_gate_v1 import (
    SEEDS,
    aggregate,
    development_gate,
    paired_bootstrap,
)


KINDS = ("primary_old", "hard_retain", "distractor", "primary_update")
CONTROLS = ("retain_all", "fixed_top", "cosine", "random_matched", "oracle")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def mean_controller(metrics):
    result = {}
    for key in ("action_accuracy", "pair_state_accuracy", "stale_deletion",
                "current_retention", "non_target_retention", "memory_fraction"):
        result[key] = statistics.fmean(row[key] for row in metrics.values())
    result["by_event_kind"] = {kind: {"accuracy": statistics.fmean(
        row["by_event_kind"][kind]["accuracy"] for row in metrics.values())}
        for kind in KINDS}
    result["by_length"] = {str(length): {key: statistics.fmean(
        row["by_length"][str(length)][key] for row in metrics.values())
        for key in ("pair_state_accuracy", "current_retention", "non_target_retention")}
        for length in LENGTHS}
    return result


def selected_controller(grid):
    candidates = []
    for epoch_key, thresholds in grid.items():
        epoch = int(epoch_key)
        for threshold_key, metric in thresholds.items():
            threshold = float(threshold_key)
            constraints = (metric["by_event_kind"]["primary_update"]["accuracy"] >= .85
                           and metric["current_retention"] >= .95
                           and metric["non_target_retention"] >= .95)
            candidates.append((int(constraints), metric["pair_state_accuracy"],
                               metric["by_event_kind"]["primary_update"]["accuracy"],
                               metric["non_target_retention"], -epoch,
                               -abs(threshold - .5), epoch, threshold))
    chosen = max(candidates)
    return chosen[-2], chosen[-1]


def selected_cosine(grid):
    candidates = []
    for threshold_key, metric in grid.items():
        threshold = float(threshold_key)
        constraints = metric["current_retention"] >= .95 and metric["non_target_retention"] >= .95
        candidates.append((int(constraints), metric["pair_state_accuracy"],
                           metric["stale_deletion"], -abs(threshold - .5), threshold))
    return max(candidates)[-1]


def audit(root, data_root, source_root):
    root, data_root, source_root = Path(root), Path(data_root), Path(source_root)
    summary = json.loads((root / "summary.json").read_text())
    training = json.loads((root / "training.json").read_text())
    manifest = json.loads((root / "training_manifest.json").read_text())
    data_manifest = json.loads((data_root / "manifest.json").read_text())

    controller_records = {
        str(seed): read_jsonl(root / f"dev_controller_seed{seed}.jsonl") for seed in SEEDS
    }
    controls = {name: read_jsonl(root / f"dev_{name}.jsonl") for name in CONTROLS}
    candidates = read_jsonl(root / "dev_candidate_rows.jsonl")
    expected_episodes = data_manifest["counts"]["dev"]["episodes"]
    expected_events = data_manifest["counts"]["dev"]["events"] - expected_episodes
    episode_sets = [{row["episode_id"] for row in rows}
                    for rows in (*controller_records.values(), *controls.values())]
    identities_ok = (all(len(rows) == expected_episodes for rows in (
        *controller_records.values(), *controls.values()))
        and all(len(values) == expected_episodes for values in episode_sets)
        and all(values == episode_sets[0] for values in episode_sets[1:]))

    raw_hashes = {path.name: sha(path) for path in root.glob("*.jsonl")}
    hash_ok = raw_hashes == training["raw_rows_sha256"]
    per_seed = {seed: aggregate(rows) for seed, rows in controller_records.items()}
    controller = mean_controller(per_seed)
    controls_aggregate = {name: aggregate(rows) for name, rows in controls.items()}
    aggregate_ok = (per_seed == summary["controller_per_seed"]
                    and controller == summary["controller"]
                    and controls_aggregate == summary["controls"])

    positive = [row for row in candidates if row["label"]]
    candidate_top1 = statistics.fmean(row["candidate_correct"] for row in positive)
    candidate_ok = (len(candidates) == expected_events
                    and len(positive) == expected_episodes
                    and all(row["event_kind"] == "primary_update" for row in positive)
                    and candidate_top1 == summary["candidate_top1"])

    primary = controller_records[str(SEEDS[0])]
    comparison = paired_bootstrap(primary, controls["random_matched"], 28201)
    comparison_ok = comparison == summary["controller_vs_random"]
    gate = development_gate(controller, candidate_top1, comparison)
    gate_ok = gate == summary["development_gate"]

    controller_deletions = sum(decision["deleted"] is not None
                               for row in primary for decision in row["decisions"])
    random_deletions = sum(decision["deleted"] is not None
                           for row in controls["random_matched"]
                           for decision in row["decisions"])
    random_ok = (controller_deletions == random_deletions
                 == training["random_matched_deletions"]
                 and all(row["decisions"][0]["deleted"] is None
                         for row in controls["random_matched"]))

    selections_ok = all(
        selected_controller(training["selections"][str(seed)]["tune_grid"])
        == (training["selections"][str(seed)]["epoch"],
            training["selections"][str(seed)]["threshold"])
        for seed in SEEDS
    ) and selected_cosine(training["cosine_grid"]) == training["cosine_threshold"]

    source_ok = (
        manifest["data_manifest_sha256"] == sha(data_root / "manifest.json")
        and manifest["source_summary_sha256"] == sha(source_root / "summary.json")
        and manifest["source_training_sha256"] == sha(source_root / "training.json")
        and manifest["source_transforms_sha256"] == sha(source_root / "transforms.pt")
        and training["checkpoint_sha256"] == sha(root / "sparse_session_gates.pt")
        and data_manifest["protocol"] == DATA_PROTOCOL
        and data_manifest["subject_disjoint"]
    )
    lock_ok = (not manifest["longmemeval_used"] and not manifest["locomo_used"]
               and not summary["longmemeval_used_for_fit_or_selection"]
               and not summary["locomo_scored"]
               and summary["next_stage"] == (
                   "preregister_locomo" if gate["passed"] else "stop_sparse_session_gate_v1")
               and len(manifest["git_commit"]) == 40)
    range_ok = all(0 <= value <= 1 for metric in (
        controller, *controls_aggregate.values()) for value in (
            metric["action_accuracy"], metric["pair_state_accuracy"],
            metric["stale_deletion"], metric["current_retention"],
            metric["non_target_retention"], metric["memory_fraction"]))
    checks = {
        "raw_hashes": hash_ok,
        "record_identities": identities_ok,
        "metric_ranges": range_ok,
        "aggregates": aggregate_ok,
        "candidate_top1": candidate_ok,
        "fit_tune_selection": selections_ok,
        "matched_random": random_ok,
        "comparison": comparison_ok,
        "development_gate": gate_ok,
        "source_and_checkpoint_hashes": source_ok,
        "claim_locks": lock_ok,
    }
    return {
        "protocol": "long_session_sparse_gate_v1_audit",
        "audit_passed": all(checks.values()),
        "checks": checks,
        "summary_sha256": sha(root / "summary.json"),
        "training_sha256": sha(root / "training.json"),
        "training_manifest_sha256": sha(root / "training_manifest.json"),
        "episodes": expected_episodes,
        "candidate_rows": len(candidates),
        "recomputed_candidate_top1": candidate_top1,
        "recomputed_gate": gate,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(args.root, args.data_root, args.source_root)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["audit_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
