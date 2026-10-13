"""Independently recompute the hierarchical candidate V2 result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from tasks.build_long_session_sparse_corpus import LENGTHS
from tasks.pack_cbf_event_warmup import sha
from tasks.train_long_session_candidate_v2 import CONTROLS, SEEDS, bootstrap, release_gate


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def aggregate(records):
    result = {
        "episodes": len(records),
        "top1": statistics.fmean(row["correct"] for row in records),
        "mrr": statistics.fmean(row["reciprocal_rank"] for row in records),
        "mean_rank": statistics.fmean(row["rank"] for row in records),
        "by_length": {},
    }
    for length in LENGTHS:
        rows = [row for row in records if row["length"] == length]
        result["by_length"][str(length)] = {
            "episodes": len(rows),
            "top1": statistics.fmean(row["correct"] for row in rows),
            "mrr": statistics.fmean(row["reciprocal_rank"] for row in rows),
            "mean_rank": statistics.fmean(row["rank"] for row in rows),
        }
    return result


def mean_metric(metrics):
    result = {key: statistics.fmean(row[key] for row in metrics.values())
              for key in ("top1", "mrr", "mean_rank")}
    result["by_length"] = {str(length): {key: statistics.fmean(
        row["by_length"][str(length)][key] for row in metrics.values())
        for key in ("top1", "mrr", "mean_rank")} for length in LENGTHS}
    return result


def selected_epoch(grid):
    rows = []
    for epoch_key, metric in grid.items():
        epoch = int(epoch_key)
        worst = min(row["top1"] for row in metric["by_length"].values())
        rows.append((metric["top1"], worst, metric["mrr"], -epoch, epoch))
    return max(rows)[-1]


def audit(root, data_root, source_root):
    root, data_root, source_root = Path(root), Path(data_root), Path(source_root)
    summary = json.loads((root / "summary.json").read_text())
    training = json.loads((root / "training.json").read_text())
    manifest = json.loads((root / "training_manifest.json").read_text())
    data_manifest = json.loads((data_root / "manifest.json").read_text())
    learned_records = {str(seed): read_jsonl(root / f"dev_learned_seed{seed}.jsonl")
                       for seed in SEEDS}
    control_records = {name: read_jsonl(root / f"dev_{name}.jsonl") for name in CONTROLS}
    expected = data_manifest["counts"]["dev"]["episodes"]
    all_records = (*learned_records.values(), *control_records.values())
    identifiers = [{row["episode_id"] for row in records} for records in all_records]
    identity_ok = (all(len(records) == expected for records in all_records)
                   and all(len(ids) == expected for ids in identifiers)
                   and all(ids == identifiers[0] for ids in identifiers[1:]))
    row_ok = all(row["rank"] >= 1 and row["rank"] <= row["candidates"]
                 and row["correct"] == int(row["rank"] == 1)
                 and row["reciprocal_rank"] == 1 / row["rank"]
                 for records in all_records for row in records)
    learned = {seed: aggregate(records) for seed, records in learned_records.items()}
    controls = {name: aggregate(records) for name, records in control_records.items()}
    mean = mean_metric(learned)
    aggregate_ok = (learned == summary["learned_per_seed"]
                    and controls == summary["controls"] and mean == summary["learned"])
    comparison = bootstrap(learned_records[str(SEEDS[0])], control_records["max_mean"], 29201)
    comparison_ok = comparison == summary["primary_vs_v1_max_cosine"]
    gate = release_gate(mean, learned, comparison)
    gate_ok = gate == summary["release_gate"]
    selection_ok = all(selected_epoch(training["selections"][str(seed)]["tune_grid"])
                       == training["selections"][str(seed)]["epoch"] for seed in SEEDS)
    raw_hashes = {path.name: sha(path) for path in root.glob("*.jsonl")}
    source_ok = (
        manifest["data_manifest_sha256"] == sha(data_root / "manifest.json")
        and manifest["source_summary_sha256"] == sha(source_root / "summary.json")
        and manifest["source_training_sha256"] == sha(source_root / "training.json")
        and manifest["source_transforms_sha256"] == sha(source_root / "transforms.pt")
        and training["checkpoint_sha256"] == sha(root / "candidate_rankers.pt")
    )
    locks_ok = (not manifest["candidate_age_feature"] and not manifest["unit_position_feature"]
                and not manifest["longmemeval_used"] and not manifest["locomo_used"]
                and not summary["longmemeval_used"] and not summary["locomo_scored"]
                and summary["next_stage"] == (
                    "preregister_sparse_gate_v2" if gate["passed"] else "stop_candidate_v2")
                and len(manifest["git_commit"]) == 40)
    checks = {
        "raw_hashes": raw_hashes == training["raw_rows_sha256"],
        "record_identities": identity_ok,
        "row_invariants": row_ok,
        "aggregates": aggregate_ok,
        "fit_tune_selection": selection_ok,
        "comparison": comparison_ok,
        "release_gate": gate_ok,
        "source_and_checkpoint_hashes": source_ok,
        "feature_and_claim_locks": locks_ok,
    }
    return {
        "protocol": "long_session_candidate_v2_audit",
        "audit_passed": all(checks.values()),
        "checks": checks,
        "summary_sha256": sha(root / "summary.json"),
        "training_sha256": sha(root / "training.json"),
        "training_manifest_sha256": sha(root / "training_manifest.json"),
        "episodes": expected,
        "recomputed_gate": gate,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True); parser.add_argument("--data-root", required=True)
    parser.add_argument("--source-root", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(); result = audit(args.root, args.data_root, args.source_root)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["audit_passed"]: raise SystemExit(1)


if __name__ == "__main__": main()
