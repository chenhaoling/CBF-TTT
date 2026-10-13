"""Independently recompute factorized causal forgetting V4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from tasks.pack_cbf_event_warmup import sha
from tasks.train_factorized_causal_forgetting_v4 import (
    GATE_SEEDS,
    aggregate,
    load_v1,
    mean_metrics,
    paired_bootstrap,
    release_gate,
)


CONTROLS = ("retain_all", "scope", "random_matched", "oracle")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def selected_gate(grid):
    rows = []
    for epoch_key, thresholds in grid.items():
        epoch = int(epoch_key)
        for threshold_key, metric in thresholds.items():
            threshold = float(threshold_key)
            safe = (metric["current_retention"] >= .98
                    and metric["non_target_retention"] >= .97
                    and metric["by_event_kind"]["hard_retain"]["accuracy"] >= .97
                    and metric["by_event_kind"]["primary_update"]["accuracy"] >= .85)
            rows.append((int(safe), metric["pair_state_accuracy"],
                         metric["by_event_kind"]["primary_update"]["accuracy"],
                         metric["non_target_retention"], -epoch,
                         -abs(threshold - .9), epoch, threshold))
    selected = max(rows); return selected[-2], selected[-1]


def selected_scope(grid):
    rows = []
    for threshold_key, metric in grid.items():
        threshold = float(threshold_key)
        safe = metric["current_retention"] >= .98 and metric["non_target_retention"] >= .97
        rows.append((int(safe), metric["pair_state_accuracy"], metric["stale_deletion"],
                     -abs(threshold - .8), threshold))
    return max(rows)[-1]


def audit(root, data_root, source_root, v1_root, v3_root):
    root, data_root, source_root, v1_root, v3_root = map(
        Path, (root, data_root, source_root, v1_root, v3_root))
    summary = json.loads((root / "summary.json").read_text())
    training = json.loads((root / "training.json").read_text())
    manifest = json.loads((root / "training_manifest.json").read_text())
    data_manifest = json.loads((data_root / "manifest.json").read_text())
    controllers = {str(seed): read_jsonl(root / f"dev_controller_seed{seed}.jsonl")
                   for seed in GATE_SEEDS}
    controls = {name: read_jsonl(root / f"dev_{name}.jsonl") for name in CONTROLS}
    candidates = read_jsonl(root / "dev_candidate_rows_seed31101.jsonl")
    v1_records, v1_hashes = load_v1(v1_root)
    expected = data_manifest["counts"]["dev"]["episodes"]
    expected_actions = data_manifest["counts"]["dev"]["events"]
    all_records = (*controllers.values(), *controls.values(), v1_records)
    ids = [{row["episode_id"] for row in rows} for rows in all_records]
    identities_ok = (all(len(rows) == expected for rows in all_records)
                     and all(len(values) == expected for values in ids)
                     and all(values == ids[0] for values in ids[1:])
                     and all(len(row["decisions"]) == row["length"]
                             for rows in all_records for row in rows))
    per_seed = {seed: aggregate(rows) for seed, rows in controllers.items()}
    controller = mean_metrics(per_seed)
    control_metrics = {name: aggregate(rows) for name, rows in controls.items()}
    aggregates_ok = (per_seed == summary["controller_per_seed"]
                     and controller == summary["controller"]
                     and control_metrics == summary["controls"])
    positives = [row for row in candidates if row["label"]]
    shortlist = statistics.fmean(row["shortlist_correct"] for row in positives)
    top1 = statistics.fmean(row["candidate_correct"] for row in positives)
    candidate_ok = (len(candidates) == expected_actions - expected
                    and len(positives) == expected
                    and shortlist == summary["shortlist_recall"]
                    and top1 == summary["candidate_top1"])
    primary = controllers[str(GATE_SEEDS[0])]
    versus_v1 = paired_bootstrap(primary, v1_records, 31301)
    versus_random = paired_bootstrap(primary, controls["random_matched"], 31302)
    comparison_ok = (versus_v1 == summary["primary_vs_v1"]
                     and versus_random == summary["primary_vs_random"])
    gate = release_gate(controller, shortlist, top1, versus_v1, versus_random)
    selections_ok = (all(selected_gate(training["selections"][str(seed)]["tune_grid"])
                         == (training["selections"][str(seed)]["epoch"],
                             training["selections"][str(seed)]["threshold"])
                         for seed in GATE_SEEDS)
                     and selected_scope(training["scope_grid"]) == training["scope_threshold"])
    primary_deletions = sum(decision["deleted"] is not None
                            for row in primary for decision in row["decisions"])
    random_deletions = sum(decision["deleted"] is not None
                           for row in controls["random_matched"] for decision in row["decisions"])
    random_ok = primary_deletions == random_deletions == training["random_matched_deletions"]
    v3_hashes = {"summary": sha(v3_root / "summary.json"),
                 "training": sha(v3_root / "training.json"),
                 "manifest": sha(v3_root / "training_manifest.json"),
                 "checkpoint": sha(v3_root / "scope_candidate_rankers.pt")}
    source_ok = (manifest["data_manifest_sha256"] == sha(data_root / "manifest.json")
                 and manifest["source_summary_sha256"] == sha(source_root / "summary.json")
                 and manifest["source_training_sha256"] == sha(source_root / "training.json")
                 and manifest["source_transforms_sha256"] == sha(source_root / "transforms.pt")
                 and training["checkpoint_sha256"] == sha(root / "factorized_sparse_gates.pt")
                 and training["v1_hashes"] == v1_hashes and training["v3_hashes"] == v3_hashes)
    locks_ok = (manifest["payload_excluded_from_candidate_ranking"]
                and not manifest["candidate_age_feature"] and not manifest["unit_position_feature"]
                and not manifest["longmemeval_used"] and not manifest["locomo_used"]
                and not summary["longmemeval_used"] and not summary["locomo_scored"]
                and summary["next_stage"] == (
                    "preregister_locomo" if gate["passed"] else "stop_factorized_gate_v4")
                and len(manifest["git_commit"]) == 40)
    raw_hashes = {path.name: sha(path) for path in root.glob("*.jsonl")}
    checks = {"raw_hashes": raw_hashes == training["raw_rows_sha256"],
              "record_identities": identities_ok, "aggregates": aggregates_ok,
              "candidate_metrics": candidate_ok, "fit_tune_selection": selections_ok,
              "matched_random": random_ok, "comparisons": comparison_ok,
              "release_gate": gate == summary["release_gate"],
              "source_v1_v3_checkpoint_hashes": source_ok,
              "feature_and_claim_locks": locks_ok}
    return {"protocol": "factorized_causal_forgetting_v4_audit",
            "audit_passed": all(checks.values()), "checks": checks,
            "summary_sha256": sha(root / "summary.json"),
            "training_sha256": sha(root / "training.json"),
            "training_manifest_sha256": sha(root / "training_manifest.json"),
            "episodes": expected, "candidate_rows": len(candidates),
            "recomputed_gate": gate}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True); parser.add_argument("--data-root", required=True)
    parser.add_argument("--source-root", required=True); parser.add_argument("--v1-root", required=True)
    parser.add_argument("--v3-root", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(); result = audit(
        args.root, args.data_root, args.source_root, args.v1_root, args.v3_root)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["audit_passed"]: raise SystemExit(1)


if __name__ == "__main__": main()
