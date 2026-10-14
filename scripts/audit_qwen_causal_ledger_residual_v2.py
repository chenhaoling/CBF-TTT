"""Independently audit the fresh scale-32 causal-ledger validation V2."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import statistics

from tasks.build_qwen_causal_ledger_residual import FAMILIES, parse_visible_block
from tasks.build_qwen_causal_ledger_residual_v2 import PHASE, RESOURCE_V2_SEED, load
from tasks.pack_cbf_event_warmup import sha
from tasks.pilot_qwen_causal_ledger_residual import assignment, semantic_apply, visible_facts
from tasks.evaluate_qwen_causal_ledger_residual_v2 import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    CONTROLS,
    PROTOCOL,
    SCALE,
)
from tasks.train_cbf_event_writer import write_json


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def close(left, right, tolerance=1e-12):
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(close(left[key], right[key], tolerance) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(close(a, b, tolerance) for a, b in zip(left, right))
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    return left == right


def bootstrap(group_values, control):
    groups = sorted(group_values)
    rng = random.Random(BOOTSTRAP_SEED + {"single_wrong": 1, "zero": 2, "random": 3}[control])
    values = []
    for _ in range(BOOTSTRAP_DRAWS):
        sample = [groups[rng.randrange(len(groups))] for _ in groups]
        values.append(statistics.fmean(group_values[group] for group in sample))
    values.sort()
    return [
        values[max(0, math.ceil(0.025 * len(values)) - 1)],
        values[min(len(values) - 1, math.ceil(0.975 * len(values)) - 1)],
    ]


def aggregate(rows):
    metrics = {}
    for name in CONTROLS:
        metrics[name] = {
            "candidate_accuracy": statistics.fmean(
                row["controls"][name]["candidate"]["prediction"] == row["answer"] for row in rows
            ),
            "truth_exact": statistics.fmean(
                row["controls"][name]["generation"]["truth_exact"] for row in rows
            ),
            "answer_nll": statistics.fmean(row["controls"][name]["answer_nll"] for row in rows),
        }
    metrics["single_wrong"]["injected_exact"] = statistics.fmean(
        row["controls"]["single_wrong"]["generation"]["injected_exact"] for row in rows
    )
    primary = "addressable_top1"
    family = {
        name: statistics.fmean(
            row["controls"][primary]["generation"]["truth_exact"] for row in rows if row["family"] == name
        ) for name in FAMILIES
    }
    role = {
        name: statistics.fmean(
            row["controls"][primary]["generation"]["truth_exact"] for row in rows if row["role"] == name
        ) for name in ("current", "historical", "anchor")
    }
    absent = statistics.fmean(
        row["controls"][primary]["generation"]["truth_exact"] for row in rows if row["absent"]
    )
    oracle = statistics.fmean(
        row["controls"][primary]["candidate"]["prediction"] == row["controls"]["exact_lookup"]["candidate"]["prediction"]
        and row["controls"][primary]["generation"]["ids"] == row["controls"]["exact_lookup"]["generation"]["ids"]
        for row in rows
    )
    twin_rows = [row for row in rows if row["twin_counterfactual"] is not None]
    twin = {
        "rows": len(twin_rows),
        "groups": len({row["group_id"] for row in twin_rows}),
        "families": sorted({row["family"] for row in twin_rows}),
        "truth_exact": statistics.fmean(row["twin_counterfactual"]["truth_exact"] for row in twin_rows),
        "twin_exact": statistics.fmean(row["twin_counterfactual"]["twin_exact"] for row in twin_rows),
    }
    intervals = {}
    for control in ("single_wrong", "zero", "random"):
        grouped = {}
        for row in rows:
            grouped.setdefault(row["group_id"], []).append(
                row["controls"][control]["answer_nll"] - row["controls"]["single_correct"]["answer_nll"]
            )
        group_values = {group: statistics.fmean(values) for group, values in grouped.items()}
        intervals[control] = {
            "mean": statistics.fmean(group_values.values()),
            "bootstrap_95ci": bootstrap(group_values, control),
        }
    checks = {
        "primary_candidate_at_least_point_95": metrics[primary]["candidate_accuracy"] >= 0.95,
        "primary_exact_at_least_point_85": metrics[primary]["truth_exact"] >= 0.85,
        "every_family_exact_at_least_point_80": min(family.values()) >= 0.80,
        "current_anchor_at_least_point_90": min(role["current"], role["anchor"]) >= 0.90,
        "historical_at_least_point_80": role["historical"] >= 0.80,
        "absent_at_least_point_85": absent >= 0.85,
        "exact_oracle_agreement": oracle == 1.0,
        "single_correct_candidate_exact_floor": metrics["single_correct"]["candidate_accuracy"] >= 0.95 and metrics["single_correct"]["truth_exact"] >= 0.90,
        "wrong_truth_ceiling_and_injected_floor": metrics["single_wrong"]["candidate_accuracy"] <= 0.10 and metrics["single_wrong"]["truth_exact"] <= 0.10 and metrics["single_wrong"]["injected_exact"] >= 0.85,
        "zero_random_candidate_at_most_point_30": max(metrics["zero"]["candidate_accuracy"], metrics["random"]["candidate_accuracy"]) <= 0.30,
        "twin_truth_ceiling_and_twin_floor": twin["rows"] > 0 and twin["truth_exact"] <= 0.10 and twin["twin_exact"] >= 0.85,
        "all_nll_gain_ci_lowers_above_zero": all(value["bootstrap_95ci"][0] > 0 for value in intervals.values()),
    }
    return {
        "metrics": metrics,
        "primary_family_exact": family,
        "primary_role_exact": role,
        "primary_absent_exact": absent,
        "oracle_agreement": oracle,
        "twin_counterfactual": twin,
        "nll_gain_intervals": intervals,
        "gate": {"checks": checks, "passed": all(checks.values())},
    }


def run(args):
    from transformers import AutoTokenizer

    root, output = Path(args.root), Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    manifest, pairs = load(root / "data")
    order, truth, twin_answers, visible_counts = {}, {}, {}, {}
    for index, (inp, target) in enumerate(pairs):
        order[inp["id"]] = index
        answers = {row["id"]: row for row in target["answers"]}
        queries = {row["id"]: row for row in inp["queries"]}
        truth[inp["id"]] = (inp, target, answers, queries)
        for row in target["answers"]:
            twin_answers[inp["group_id"], target["twin"], row["id"]] = row["answer"]
        state = []
        for block in inp["blocks"]:
            event = assignment(parse_visible_block(block["text"], PHASE))
            state = semantic_apply(state, event, block["index"])
            visible_counts[inp["id"], block["index"]] = len(visible_facts(state))

    rows, resources, hashes = [], [], {}
    shard_integrity = True
    for shard in (0, 1):
        row_path = root / f"v2_shard{shard}.jsonl"
        resource_path = root / f"v2_resource_shard{shard}.json"
        part, resource = read_jsonl(row_path), json.loads(resource_path.read_text())
        hashes[row_path.name], hashes[resource_path.name] = sha(row_path), sha(resource_path)
        shard_integrity &= all(row["shard"] == shard and order[row["episode_id"]] % 2 == shard for row in part)
        shard_integrity &= all((
            resource["protocol"] == PROTOCOL,
            resource["episodes"] == 56,
            resource["rows"] == len(part) == 504,
            resource["rows_sha256"] == hashes[row_path.name],
            resource["data_manifest_sha256"] == sha(root / "data/manifest.json"),
            resource["scale"] == SCALE,
            resource["seed"] == RESOURCE_V2_SEED,
            not resource["loading_info"]["missing_keys"],
            not resource["loading_info"]["unexpected_keys"],
            not resource["loading_info"]["mismatched_keys"],
            not resource["development_generated"],
            not resource["confirmation_generated"],
            not resource["ood_generated"],
        ))
        rows.extend(part); resources.append(resource)
    coverage = (
        len(rows) == 1008
        and len({(row["episode_id"], row["query_id"]) for row in rows}) == 1008
        and {row["episode_id"] for row in rows} == set(truth)
    )
    row_integrity = True
    for row in rows:
        inp, target, answers, queries = truth[row["episode_id"]]
        answer, query = answers[row["query_id"]], queries[row["query_id"]]
        answer_ids = tokenizer.encode(answer["answer"], add_special_tokens=False)
        wrong_ids = tokenizer.encode(row["wrong_value"], add_special_tokens=False)
        other = twin_answers[inp["group_id"], 1 - target["twin"], row["query_id"]]
        row_integrity &= all((
            row["protocol"] == PROTOCOL,
            row["group_id"] == inp["group_id"],
            row["family"] == target["family"],
            row["role"] == query["role"] == answer["role"],
            row["absent"] == answer["absent"],
            row["answer"] == answer["answer"],
            row["candidates"] == query["candidates"],
            len(answer_ids) == len(wrong_ids) == 5,
            row["visible_facts"] == visible_counts[row["episode_id"], row["block_index"]],
            set(row["controls"]) == set(CONTROLS),
            row["twin_causally_equal"] == (other == answer["answer"]),
            (row["twin_counterfactual"] is None) == (other == answer["answer"]),
        ))
        for name, control in row["controls"].items():
            candidate = control["candidate"]
            prediction = min(candidate["scores"], key=lambda value: (candidate["scores"][value], value))
            generation = control["generation"]
            row_integrity &= all((
                set(candidate["scores"]) == set(query["candidates"]),
                candidate["prediction"] == prediction,
                close(control["answer_nll"], candidate["scores"][answer["answer"]]),
                len(generation["ids"]) == 5,
                generation["truth_exact"] == int(generation["ids"] == answer_ids),
                math.isfinite(control["answer_nll"]),
            ))
        row_integrity &= row["controls"]["single_wrong"]["generation"]["injected_exact"] == int(
            row["controls"]["single_wrong"]["generation"]["ids"] == wrong_ids
        )
        if row["twin_counterfactual"] is not None:
            twin_ids = tokenizer.encode(other, add_special_tokens=False)
            twin = row["twin_counterfactual"]
            row_integrity &= all((
                twin["answer"] == other,
                twin["truth_exact"] == int(twin["generation"]["ids"] == answer_ids),
                twin["twin_exact"] == int(twin["generation"]["ids"] == twin_ids),
            ))

    summary = json.loads((root / "summary.json").read_text())
    recomputed = aggregate(rows)
    producer = {key: summary[key] for key in recomputed}
    aggregate_integrity = close(producer, recomputed)
    terminal = (
        (root / "terminal_status.txt").read_text().strip() == "completed_qwen_causal_ledger_residual_v2"
        and (root / "runner_exit_code.txt").read_text().strip() == "0"
    )
    sealed = all(
        not summary[name] and not manifest[name]
        for name in ("development_generated", "confirmation_generated", "ood_generated")
    )
    checks = {
        "terminal_status_and_exit": terminal,
        "fresh_source_and_coverage": coverage,
        "shard_hashes_model_and_resources": shard_integrity,
        "row_state_score_generation_and_twin": row_integrity,
        "independent_aggregates_bootstrap_and_gate": aggregate_integrity,
        "downstream_splits_sealed": sealed,
    }
    result = {
        "protocol": "qwen_causal_ledger_residual_v2_audit_v1",
        "passed": all(checks.values()),
        "checks": checks,
        "hashes": hashes,
        "data_manifest_sha256": sha(root / "data/manifest.json"),
        "summary_sha256": sha(root / "summary.json"),
        "recomputed": recomputed,
    }
    write_json(output, result)
    if not result["passed"]:
        raise RuntimeError("causal-ledger V2 validation audit failed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
