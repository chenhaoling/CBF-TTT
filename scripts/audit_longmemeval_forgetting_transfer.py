"""Independently recompute the compact LongMemEval-S transfer result."""

import argparse
import json
from pathlib import Path
import random
import statistics

from tasks.pack_cbf_event_warmup import sha


POLICIES = ("retain_all", "window8", "fixed_top", "random_matched",
            "frozen_controller", "oracle_pair")
METRICS = ("pair_state_correct", "latest_retained", "stale_deleted",
           "nonanswer_retention", "memory_fraction", "current_hit1", "current_hit5",
           "stale_hit1", "stale_hit5", "latest_over_stale")


def bootstrap(rows, left, right, metric, seed, direction=1., draws=10000):
    rng = random.Random(seed); values = []
    for _ in range(draws):
        chosen = [rows[rng.randrange(len(rows))] for _ in rows]
        values.append(direction * statistics.fmean(
            row["policy_metrics"][left][metric] - row["policy_metrics"][right][metric]
            for row in chosen))
    values.sort(); return [values[int(.025 * draws)], values[int(.975 * draws) - 1]]


def audit(root):
    root = Path(root); summary = json.loads((root / "summary.json").read_text())
    training = json.loads((root / "training.json").read_text()); rows = []; shard_checks = []
    for shard in range(2):
        path = root / f"shard_{shard}.jsonl"
        meta = json.loads((root / f"shard_{shard}.jsonl.audit.json").read_text())
        shard_checks.append(meta["rows_sha256"] == sha(path)
                            and training["raw_rows_sha256"][path.name] == sha(path)
                            and meta["questions"] == 39 and meta["shard"] == shard)
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line)
    rows.sort(key=lambda row: row["question_id"])
    identity_ok = len(rows) == len({row["question_id"] for row in rows}) == 78
    range_ok = all(set(row["policy_metrics"]) == set(POLICIES)
                   and row["encoded_windows"] > 0
                   and all(0 <= row["policy_metrics"][policy][metric] <= 1
                           for policy in POLICIES for metric in METRICS)
                   for row in rows)
    recomputed = {policy: {metric: statistics.fmean(
        row["policy_metrics"][policy][metric] for row in rows) for metric in METRICS}
        for policy in POLICIES}
    aggregate_ok = recomputed == summary["policies"]
    comparisons = {
        "pair_state_vs_random": {
            "gain": recomputed["frozen_controller"]["pair_state_correct"]
                    - recomputed["random_matched"]["pair_state_correct"],
            "bootstrap_95ci": bootstrap(rows, "frozen_controller", "random_matched",
                                          "pair_state_correct", 27101)},
        "current_hit5_vs_retain": {
            "gain": recomputed["frozen_controller"]["current_hit5"]
                    - recomputed["retain_all"]["current_hit5"],
            "bootstrap_95ci": bootstrap(rows, "frozen_controller", "retain_all",
                                          "current_hit5", 27102)},
        "stale_hit5_reduction_vs_retain": {
            "gain": recomputed["retain_all"]["stale_hit5"]
                    - recomputed["frozen_controller"]["stale_hit5"],
            "bootstrap_95ci": bootstrap(rows, "frozen_controller", "retain_all",
                                          "stale_hit5", 27103, -1.)},
    }
    comparison_ok = comparisons == summary["comparisons"]
    checks = {
        "pair_state_beats_random": comparisons["pair_state_vs_random"]["bootstrap_95ci"][0] > 0,
        "current_hit5_noninferior": comparisons["current_hit5_vs_retain"]["bootstrap_95ci"][0] > -.02,
        "stale_hit5_reduced": comparisons["stale_hit5_reduction_vs_retain"]["bootstrap_95ci"][0] > 0,
        "latest_retention_at_least_95": recomputed["frozen_controller"]["latest_retained"] >= .95,
        "nonanswer_retention_at_least_80": recomputed["frozen_controller"]["nonanswer_retention"] >= .80,
    }
    gate_ok = summary["transfer_gate"] == {"checks": checks, "passed": all(checks.values())}
    lock_ok = (not summary["official_longmemeval_score"] and not summary["locomo_scored"]
               and summary["next_stage"] == "stop_longmemeval_zero_shot_transfer")
    all_checks = {"shard_hashes": all(shard_checks), "identities": identity_ok,
                  "metric_ranges": range_ok, "aggregates": aggregate_ok,
                  "comparisons": comparison_ok, "gate": gate_ok,
                  "claim_locks": lock_ok,
                  "terminal_marker": (root / "terminal_status.txt").read_text().strip()
                                     == summary["terminal_status"]}
    return {"protocol": "longmemeval_forgetting_transfer_v1_audit",
            "audit_passed": all(all_checks.values()), "checks": all_checks,
            "summary_sha256": sha(root / "summary.json"),
            "training_sha256": sha(root / "training.json"),
            "questions": len(rows), "recomputed_gate": {"checks": checks,
                                                           "passed": all(checks.values())}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(); result = audit(args.root)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
