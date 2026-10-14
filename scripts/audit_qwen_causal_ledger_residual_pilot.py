"""Independently audit the frozen-Qwen causal-ledger resource/content pilot."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import statistics

from tasks.build_qwen_causal_ledger_residual import (
    FAMILIES,
    PILOT_SEED,
    load_pilot,
    parse_visible_block,
    validate,
)
from tasks.pack_cbf_event_warmup import sha
from tasks.pilot_qwen_causal_ledger_residual import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    FORMAL_CHOICES,
    PRIMARY_SCALE,
    PROTOCOL,
    SCALES,
)
from tasks.pilot_qwen_dynamic_record_forgetting import ACTIONS, valid_actions
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


def normalize(value):
    return " ".join(value.strip().casefold().split())


def independent_label(outcomes, actions):
    key = str(PRIMARY_SCALE)
    ranks = []
    for index, (outcome, action) in enumerate(zip(outcomes, actions)):
        row = outcome[key]
        rank = (
            round(row["accuracy"] / 1e-12) * 1e-12,
            -round(row["nll"] / 1e-6) * 1e-6,
            -row["removed_mass"],
            -ACTIONS.index((action[0], tuple(action[1]))),
        )
        ranks.append((rank, index))
    return max(ranks)[1]


def bootstrap(group_values, control):
    groups = sorted(group_values)
    rng = random.Random(BOOTSTRAP_SEED + {"wrong": 1, "zero": 2, "random": 3}[control])
    draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        sample = [groups[rng.randrange(len(groups))] for _ in groups]
        draws.append(statistics.fmean(group_values[group] for group in sample))
    draws.sort()
    return [
        draws[max(0, math.ceil(0.025 * len(draws)) - 1)],
        draws[min(len(draws) - 1, math.ceil(0.975 * len(draws)) - 1)],
    ]


def aggregate(content, labels, resources):
    scale_table = {}
    for scale in SCALES:
        key = str(scale)
        scale_table[key] = {}
        for name in ("correct", "wrong", "zero", "random"):
            scale_table[key][name] = {
                "candidate_accuracy": statistics.fmean(
                    row["controls"][name]["by_scale"][key]["prediction"] == row["answer"]
                    for row in content
                ),
                "answer_nll": statistics.fmean(
                    row["controls"][name]["answer_nll"][key] for row in content
                ),
            }
        scale_table[key]["correct"]["greedy_exact"] = statistics.fmean(
            row["correct_generation"][key]["normalized_exact"] for row in content
        )
        scale_table[key]["correct"]["eos_rate"] = statistics.fmean(
            row["correct_generation"][key]["stop_reason"] == "eos" for row in content
        )
    intervals = {}
    for control in ("wrong", "zero", "random"):
        grouped = {}
        for row in content:
            key = str(PRIMARY_SCALE)
            grouped.setdefault(row["group_id"], []).append(
                row["controls"][control]["answer_nll"][key]
                - row["controls"]["correct"]["answer_nll"][key]
            )
        group_values = {group: statistics.fmean(values) for group, values in grouped.items()}
        intervals[control] = {
            "mean": statistics.fmean(group_values.values()),
            "bootstrap_95ci": bootstrap(group_values, control),
        }
    family_accuracy = {
        family: statistics.fmean(
            row["controls"]["correct"]["by_scale"][str(PRIMARY_SCALE)]["prediction"]
            == row["answer"]
            for row in content if row["family"] == family
        ) for family in FAMILIES
    }
    primary = scale_table[str(PRIMARY_SCALE)]
    sensitivity = all(
        scale_table[str(scale)]["correct"]["candidate_accuracy"]
        >= primary["correct"]["candidate_accuracy"] - 0.05
        and scale_table[str(scale)]["correct"]["eos_rate"]
        <= primary["correct"]["eos_rate"] + 0.05
        for scale in (4.0, 16.0)
    )
    checks = {
        "correct_candidate_at_least_point_95": primary["correct"]["candidate_accuracy"] >= 0.95,
        "correct_greedy_at_least_point_85": primary["correct"]["greedy_exact"] >= 0.85,
        "wrong_candidate_at_most_point_10": primary["wrong"]["candidate_accuracy"] <= 0.10,
        "zero_candidate_at_most_point_30": primary["zero"]["candidate_accuracy"] <= 0.30,
        "all_nll_gain_ci_lowers_above_zero": all(row["bootstrap_95ci"][0] > 0 for row in intervals.values()),
        "every_family_correct_at_least_point_90": min(family_accuracy.values()) >= 0.90,
        "scale_sensitivity_within_five_points": sensitivity,
    }
    wall = max(row["seconds"] for row in resources)
    allocated = max(row["overall_peak_allocated_gib"] for row in resources)
    estimates = {str(choice): wall * choice / 8.0 for choice in FORMAL_CHOICES}
    eligible = [
        choice for choice in FORMAL_CHOICES
        if all(checks.values()) and allocated < 24.0 and estimates[str(choice)] < 6 * 3600
    ]
    return {
        "content": scale_table,
        "nll_gain_intervals": intervals,
        "family_correct_accuracy": family_accuracy,
        "content_gate": {"checks": checks, "passed": all(checks.values())},
        "labels": {
            "rows": len(labels),
            "mean_seconds": statistics.fmean(row["label_seconds"] for row in labels),
            "p95_seconds": sorted(row["label_seconds"] for row in labels)[math.ceil(0.95 * len(labels)) - 1],
            "mean_seconds_per_action": statistics.fmean(row["seconds_per_action"] for row in labels),
            "valid_action_counts": dict(Counter(str(row["valid_action_count"]) for row in labels)),
            "label_counts": dict(Counter(str(row["label"]) for row in labels)),
        },
        "selected_groups_per_family": max(eligible) if eligible else None,
        "estimated_development_seconds": estimates,
    }


def run(args):
    root, output = Path(args.root), Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    manifest, pairs = load_pilot(root / "data")
    construction = validate(pairs, expected_groups=8)
    truth = {}
    pair_order = {}
    for index, (inp, target) in enumerate(pairs):
        truth[inp["id"]] = {
            "input": inp,
            "target": target,
            "answers": {row["id"]: row for row in target["answers"]},
            "queries": {row["id"]: row for row in inp["queries"]},
        }
        pair_order[inp["id"]] = index

    labels, content, resources, hashes = [], [], [], {}
    hashes["data/manifest.json"] = sha(root / "data/manifest.json")
    shard_integrity = True
    for shard in (0, 1):
        label_path = root / f"labels_shard{shard}.jsonl"
        content_path = root / f"content_shard{shard}.jsonl"
        resource_path = root / f"resource_shard{shard}.json"
        part_labels, part_content = read_jsonl(label_path), read_jsonl(content_path)
        resource = json.loads(resource_path.read_text())
        hashes[label_path.name], hashes[content_path.name] = sha(label_path), sha(content_path)
        hashes[resource_path.name] = sha(resource_path)
        shard_integrity &= all(
            row["shard"] == shard and pair_order[row["episode_id"]] % 2 == shard
            for row in part_labels + part_content
        )
        shard_integrity &= (
            resource["protocol"] == PROTOCOL
            and resource["shard"] == shard
            and resource["num_shards"] == 2
            and resource["pairs"] == 56
            and resource["labels"] == len(part_labels) == 168
            and resource["content_rows"] == len(part_content) == 504
            and resource["labels_sha256"] == hashes[label_path.name]
            and resource["content_sha256"] == hashes[content_path.name]
            and resource["data_manifest_sha256"] == hashes["data/manifest.json"]
            and not resource["loading_info"]["missing_keys"]
            and not resource["loading_info"]["unexpected_keys"]
            and not resource["loading_info"]["mismatched_keys"]
            and not resource["development_generated"]
            and not resource["confirmation_generated"]
            and not resource["ood_generated"]
        )
        labels.extend(part_labels); content.extend(part_content); resources.append(resource)

    coverage = (
        manifest["seed"] == PILOT_SEED
        and construction["episodes"] == 112
        and len(labels) == 336
        and len(content) == 1008
        and len({(row["episode_id"], row["block_index"]) for row in labels}) == 336
        and len({(row["episode_id"], row["query_id"]) for row in content}) == 1008
        and {row["episode_id"] for row in labels} == set(truth)
        and {row["episode_id"] for row in content} == set(truth)
    )

    label_integrity = True
    for row in labels:
        source = truth[row["episode_id"]]
        block = source["input"]["blocks"][row["block_index"] - 1]
        event = parse_visible_block(block["text"], "pilot")
        actions = [[clear, list(retention)] for clear, retention in valid_actions(event, tuple(range(len(row["slots"]))))]
        selected = independent_label(row["outcomes"], row["actions"])
        label_integrity &= all((
            row["protocol"] == PROTOCOL,
            row["family"] == source["target"]["family"],
            row["block_index"] in source["input"]["decision_boundaries"],
            row["actions"] == actions,
            row["valid_action_count"] == len(actions),
            len(row["outcomes"]) == len(actions),
            row["label_index"] == selected,
            row["label"] == actions[selected],
            row["label_seconds"] > 0,
            close(row["seconds_per_action"], row["label_seconds"] / len(actions)),
            row["answer_tokens"] == 5,
            row["candidate_count"] == 6,
            row["peak_allocated_gib"] > 0,
            row["peak_reserved_gib"] >= row["peak_allocated_gib"],
        ))
        for outcome in row["outcomes"]:
            label_integrity &= set(outcome) == {str(scale) for scale in SCALES}
            label_integrity &= all(
                0 <= item["accuracy"] <= 1
                and math.isfinite(item["nll"])
                and item["removed_mass"] >= 0
                for item in outcome.values()
            )

    content_integrity = True
    for row in content:
        source = truth[row["episode_id"]]
        answer, query = source["answers"][row["query_id"]], source["queries"][row["query_id"]]
        content_integrity &= all((
            row["protocol"] == PROTOCOL,
            row["group_id"] == source["input"]["group_id"],
            row["family"] == source["target"]["family"],
            row["block_index"] == query["after_block"],
            row["role"] == query["role"] == answer["role"],
            row["absent"] == answer["absent"],
            row["answer"] == answer["answer"],
            row["candidates"] == query["candidates"],
            set(row["controls"]) == {"correct", "wrong", "zero", "random"},
            set(row["correct_generation"]) == {str(scale) for scale in SCALES},
        ))
        for name, control in row["controls"].items():
            for scale in SCALES:
                key = str(scale); scored = control["by_scale"][key]
                content_integrity &= (
                    set(scored["scores"]) == set(query["candidates"])
                    and scored["prediction"] in query["candidates"]
                    and close(control["answer_nll"][key], scored["scores"][answer["answer"]])
                    and all(math.isfinite(value) for value in scored["scores"].values())
                )
        for scale in SCALES:
            generation = row["correct_generation"][str(scale)]
            content_integrity &= generation["normalized_exact"] == int(
                normalize(generation["text"]) == normalize(answer["answer"])
            )

    summary = json.loads((root / "summary.json").read_text())
    recomputed = aggregate(content, labels, resources)
    producer_subset = {key: summary[key] for key in (
        "content", "nll_gain_intervals", "family_correct_accuracy", "content_gate", "labels"
    )}
    aggregate_integrity = close(
        producer_subset,
        {key: recomputed[key] for key in producer_subset},
    ) and close(
        summary["resources"]["estimated_development_seconds"],
        recomputed["estimated_development_seconds"],
    ) and summary["resources"]["selected_groups_per_family"] == recomputed["selected_groups_per_family"]

    terminal = (
        (root / "terminal_status.txt").read_text().strip() == "completed_qwen_causal_ledger_residual_pilot"
        and (root / "runner_exit_code.txt").read_text().strip() == "0"
        and summary["terminal_status"] == "completed_qwen_causal_ledger_residual_pilot"
    )
    sealed = all(
        not summary[name] and not manifest[name]
        for name in ("development_generated", "confirmation_generated", "ood_generated")
    )
    checks = {
        "terminal_status_and_exit": terminal,
        "construction_and_coverage": coverage,
        "shard_hashes_and_resources": shard_integrity,
        "label_actions_and_selection": label_integrity,
        "content_rows_and_generations": content_integrity,
        "independent_aggregates_and_gate": aggregate_integrity,
        "downstream_splits_sealed": sealed,
    }
    result = {
        "protocol": "qwen_causal_ledger_residual_pilot_audit_v1",
        "passed": all(checks.values()),
        "checks": checks,
        "hashes": hashes,
        "recomputed": recomputed,
        "producer_summary_sha256": sha(root / "summary.json"),
    }
    write_json(output, result)
    if not result["passed"]:
        raise RuntimeError("causal-ledger residual pilot audit failed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
