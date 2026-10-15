"""Evaluate the one-time scale-48 causal-ledger lifecycle OOD split."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import statistics

from tasks.build_qwen_causal_ledger_ood_v1 import (
    GROUPS_PER_SUITE,
    PHASE,
    SUITES,
    load,
)
from tasks.evaluate_qwen_causal_ledger_development_v2 import (
    ORACLE,
    POLICIES,
    PRIMARY,
    collect,
)
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "qwen_causal_ledger_ood_v1"
SCALE = 48.0
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 202610162
EXPECTED_EPISODES = GROUPS_PER_SUITE * len(SUITES) * 2
EXPECTED_ROWS = EXPECTED_EPISODES * 9


def _bounds(values):
    values = sorted(values)
    return [
        values[max(0, math.ceil(0.025 * len(values)) - 1)],
        values[min(len(values) - 1, math.ceil(0.975 * len(values)) - 1)],
    ]


def bootstrap(primary, comparator):
    groups = sorted(primary)
    rng = random.Random(BOOTSTRAP_SEED)
    exact, nll = [], []
    for _ in range(BOOTSTRAP_DRAWS):
        sample = [groups[rng.randrange(len(groups))] for _ in groups]
        exact.append(statistics.fmean(
            primary[group]["exact"] - comparator[group]["exact"] for group in sample
        ))
        nll.append(statistics.fmean(
            comparator[group]["nll"] - primary[group]["nll"] for group in sample
        ))
    return {
        "exact_gain": statistics.fmean(
            primary[group]["exact"] - comparator[group]["exact"] for group in groups
        ),
        "exact_bootstrap_95ci": _bounds(exact),
        "nll_gain": statistics.fmean(
            comparator[group]["nll"] - primary[group]["nll"] for group in groups
        ),
        "nll_bootstrap_95ci": _bounds(nll),
    }


def aggregate(rows):
    groups = sorted({row["group_id"] for row in rows})
    lengths = sorted({row["length"] for row in rows})
    metrics, group_metrics = {}, {}
    for policy in POLICIES:
        metrics[policy] = {
            "candidate_accuracy": statistics.fmean(
                row["policies"][policy]["candidate_correct"] for row in rows
            ),
            "exact": statistics.fmean(row["policies"][policy]["exact"] for row in rows),
            "nll": statistics.fmean(row["policies"][policy]["nll"] for row in rows),
            "roles": {
                role: statistics.fmean(
                    row["policies"][policy]["exact"] for row in rows
                    if row["role"] == role
                )
                for role in ("current", "historical", "anchor")
            },
            "suites": {
                suite: {
                    "candidate_accuracy": statistics.fmean(
                        row["policies"][policy]["candidate_correct"] for row in rows
                        if row["family"] == suite
                    ),
                    "exact": statistics.fmean(
                        row["policies"][policy]["exact"] for row in rows
                        if row["family"] == suite
                    ),
                    "nll": statistics.fmean(
                        row["policies"][policy]["nll"] for row in rows
                        if row["family"] == suite
                    ),
                }
                for suite in SUITES
            },
            "lengths": {
                str(length): statistics.fmean(
                    row["policies"][policy]["exact"] for row in rows
                    if row["length"] == length
                )
                for length in lengths
            },
            "absent": statistics.fmean(
                row["policies"][policy]["exact"] for row in rows if row["absent"]
            ),
        }
        group_metrics[policy] = {}
        for group in groups:
            part = [row for row in rows if row["group_id"] == group]
            group_metrics[policy][group] = {
                "exact": statistics.fmean(row["policies"][policy]["exact"] for row in part),
                "nll": statistics.fmean(row["policies"][policy]["nll"] for row in part),
            }
    comparators = [name for name in POLICIES if name not in {PRIMARY, ORACLE}]
    strongest = max(comparators, key=lambda name: metrics[name]["exact"])
    comparison = bootstrap(group_metrics[PRIMARY], group_metrics[strongest])
    oracle_agreement = statistics.fmean(
        row["policies"][PRIMARY]["candidate_prediction"]
        == row["policies"][ORACLE]["candidate_prediction"]
        and row["policies"][PRIMARY]["generated_ids"]
        == row["policies"][ORACLE]["generated_ids"]
        for row in rows
    )
    primary = metrics[PRIMARY]
    checks = {
        "primary_macro_candidate_at_least_point_85": primary["candidate_accuracy"] >= 0.85,
        "every_suite_candidate_at_least_point_75": min(
            row["candidate_accuracy"] for row in primary["suites"].values()
        ) >= 0.75,
        "primary_exact_at_least_point_80": primary["exact"] >= 0.80,
        "current_anchor_at_least_point_90": min(
            primary["roles"]["current"], primary["roles"]["anchor"]
        ) >= 0.90,
        "historical_at_least_point_80": primary["roles"]["historical"] >= 0.80,
        "absent_at_least_point_85": primary["absent"] >= 0.85,
        "gain_at_least_point_03": comparison["exact_gain"] >= 0.03,
        "gain_ci_lower_above_zero": comparison["exact_bootstrap_95ci"][0] > 0,
        "nll_gain_ci_lower_above_zero": comparison["nll_bootstrap_95ci"][0] > 0,
        "exact_oracle_agreement": oracle_agreement == 1.0,
    }
    return {
        "metrics": metrics,
        "strongest_non_oracle_comparator": strongest,
        "comparison": comparison,
        "oracle_agreement": oracle_agreement,
        "gate": {"checks": checks, "passed": all(checks.values())},
    }


def summarize(args):
    root, output = Path(args.root), Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    rows, resources = [], []
    for shard in (0, 1):
        path = root / f"ood_shard{shard}.jsonl"
        resource = json.loads((root / f"ood_resource_shard{shard}.json").read_text())
        part = [json.loads(line) for line in path.read_text().splitlines() if line]
        if (
            resource["rows_sha256"] != sha(path) or resource["rows"] != len(part)
            or resource["episodes"] != EXPECTED_EPISODES // 2
            or len(part) != EXPECTED_ROWS // 2
        ):
            raise ValueError("causal-ledger OOD shard mismatch")
        rows.extend(part); resources.append(resource)
    if len(rows) != EXPECTED_ROWS or len({
        (row["episode_id"], row["query_id"]) for row in rows
    }) != EXPECTED_ROWS:
        raise ValueError("causal-ledger OOD coverage mismatch")
    result = {
        "protocol": PROTOCOL,
        "scale": SCALE,
        "rows": len(rows),
        **aggregate(rows),
        "resources": {
            "two_gpu_critical_path_seconds": max(row["seconds"] for row in resources),
            "peak_allocated_gib": max(row["peak_allocated_gib"] for row in resources),
            "peak_reserved_gib": max(row["peak_reserved_gib"] for row in resources),
            "shards": resources,
        },
        "confirmation_generated": True,
        "ood_generated": True,
    }
    result["next_stage"] = (
        "natural_text_scope_and_lifecycle_transfer"
        if result["gate"]["passed"] else "stop_and_diagnose_ood_failure"
    )
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument("--data-root", required=True)
    collect_parser.add_argument("--model", required=True)
    collect_parser.add_argument("--tokenizer", required=True)
    collect_parser.add_argument("--shard", type=int, required=True)
    collect_parser.add_argument("--num-shards", type=int, default=2)
    collect_parser.add_argument("--output", required=True)
    collect_parser.add_argument("--resource-output", required=True)
    summary_parser = sub.add_parser("summarize")
    summary_parser.add_argument("--root", required=True)
    summary_parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "collect":
        result = collect(
            args, scale=SCALE, protocol=PROTOCOL, loader=load, phase=PHASE,
            confirmation_generated=True, ood_generated=True,
        )
    else:
        result = summarize(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
