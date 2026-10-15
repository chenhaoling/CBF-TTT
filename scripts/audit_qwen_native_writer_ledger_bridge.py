"""Independently audit the single-record native-writer ledger bridge pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

from tasks.pack_cbf_event_warmup import sha
from tasks.pilot_qwen_causal_ledger_residual import render_query
from tasks.pilot_qwen_native_writer_ledger_bridge import (
    ARMS,
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    CHUNK_SIZE,
    PROTOCOL,
    fill_chunk,
    select_samples,
)
from tasks.train_cbf_event_writer import write_json


def token_hash(ids):
    return hashlib.sha256(b"".join(
        int(value).to_bytes(4, "little") for value in ids
    )).hexdigest()


def bootstrap(rows, control):
    rng = random.Random(BOOTSTRAP_SEED + {"wrong_writer": 1, "zero": 2}[control])
    gains = [
        row["arms"][control]["scores"][row["answer"]]
        - row["arms"]["correct_writer"]["scores"][row["answer"]]
        for row in rows
    ]
    values = []
    for _ in range(BOOTSTRAP_DRAWS):
        values.append(statistics.fmean(gains[rng.randrange(len(gains))] for _ in gains))
    values.sort()
    return [
        values[max(0, math.ceil(.025 * len(values)) - 1)],
        values[min(len(values) - 1, math.ceil(.975 * len(values)) - 1)],
    ]


def recompute(rows):
    metrics = {}
    for arm in ARMS:
        metrics[arm] = {
            "candidate_accuracy": statistics.fmean(
                row["arms"][arm]["prediction"] == row["answer"] for row in rows
            ),
            "exact": statistics.fmean(row["arms"][arm]["exact"] for row in rows),
            "answer_nll": statistics.fmean(
                row["arms"][arm]["scores"][row["answer"]] for row in rows
            ),
        }
    metrics["wrong_writer"]["injected_wrong_accuracy"] = statistics.fmean(
        row["arms"]["wrong_writer"]["prediction"] == row["wrong"] for row in rows
    )
    intervals = {name: bootstrap(rows, name) for name in ("wrong_writer", "zero")}
    checks = {
        "correct_candidate_at_least_point_80": metrics["correct_writer"]["candidate_accuracy"] >= .80,
        "correct_exact_at_least_point_60": metrics["correct_writer"]["exact"] >= .60,
        "wrong_truth_at_most_point_20": metrics["wrong_writer"]["candidate_accuracy"] <= .20,
        "wrong_injected_at_least_point_80": metrics["wrong_writer"]["injected_wrong_accuracy"] >= .80,
        "zero_truth_at_most_point_30": metrics["zero"]["candidate_accuracy"] <= .30,
        "correct_nll_lower_bounds_positive": min(value[0] for value in intervals.values()) > 0,
        "external_candidate_at_least_point_95": metrics["external_residual"]["candidate_accuracy"] >= .95,
        "external_exact_at_least_point_85": metrics["external_residual"]["exact"] >= .85,
    }
    return {
        "metrics": metrics,
        "correct_nll_gain_bootstrap_95ci": intervals,
        "gate": {"checks": checks, "passed": all(checks.values())},
    }


def close(left, right, tolerance=1e-12):
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            close(left[key], right[key], tolerance) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            close(a, b, tolerance) for a, b in zip(left, right)
        )
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=0, abs_tol=tolerance)
    return left == right


def main():
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--confirmation-root", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root, confirmation_root, output = map(
        Path, (args.root, args.confirmation_root, args.output)
    )
    if output.exists():
        raise FileExistsError(output)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    expected = {row["group_id"]: row for row in select_samples()}
    rows, resources, hashes = [], [], {}
    shard_ok = True
    for shard in (0, 1):
        path = root / f"bridge_shard{shard}.jsonl"
        resource_path = root / f"bridge_resource_shard{shard}.json"
        part = [json.loads(line) for line in path.read_text().splitlines() if line]
        resource = json.loads(resource_path.read_text())
        hashes[path.name] = sha(path); hashes[resource_path.name] = sha(resource_path)
        shard_ok &= all((
            resource["protocol"] == PROTOCOL,
            resource["shard"] == shard,
            resource["rows"] == len(part) == 7,
            resource["rows_sha256"] == hashes[path.name],
            resource["chunk_size"] == CHUNK_SIZE,
            len(resource["ttt_layers"]) == 7,
            resource["ttt_lr"] == .3,
            not resource["loading_info"]["missing_keys"],
            not resource["loading_info"]["unexpected_keys"],
            not resource["loading_info"]["mismatched_keys"],
            not resource["confirmation_scored"],
            not resource["natural_benchmark_scored"],
            bool(resource["source_files_sha256"]),
        ))
        shard_ok &= all(row["shard"] == shard for row in part)
        rows.extend(part); resources.append(resource)
    shard_ok &= resources[0]["source_files_sha256"] == resources[1]["source_files_sha256"]
    coverage = (
        len(rows) == len(expected) == 14
        and set(expected) == {row["group_id"] for row in rows}
    )
    row_ok = True
    for row in rows:
        source = expected[row["group_id"]]
        query = source["query"]
        answer_ids = tokenizer.encode(source["answer"], add_special_tokens=False)
        correct_ids = fill_chunk(tokenizer, query["key"], source["answer"])
        wrong_ids = fill_chunk(tokenizer, query["key"], source["wrong"])
        row_ok &= all((
            row["protocol"] == PROTOCOL,
            row["family"] == source["family"],
            row["group_index"] == source["group_index"],
            row["query_id"] == query["id"],
            row["query_key"] == query["key"],
            row["answer"] == source["answer"],
            row["wrong"] == source["wrong"],
            row["candidates"] == query["candidates"],
            row["answer_ids"] == answer_ids,
            row["chunk_tokens"] == len(correct_ids) == len(wrong_ids) == CHUNK_SIZE,
            row["correct_chunk_sha256"] == token_hash(correct_ids),
            row["wrong_chunk_sha256"] == token_hash(wrong_ids),
            set(row["arms"]) == set(ARMS),
            row["memory"]["correct"]["layers"] == 7,
            row["memory"]["wrong"]["layers"] == 7,
        ))
        for arm, value in row["arms"].items():
            row_ok &= all((
                set(value["scores"]) == set(query["candidates"]),
                value["prediction"] in query["candidates"],
                len(value["generated_ids"]) == 5,
                value["exact"] == int(value["generated_ids"] == answer_ids),
                all(math.isfinite(score) for score in value["scores"].values()),
            ))
    summary = json.loads((root / "summary.json").read_text())
    recomputed = recompute(rows)
    aggregate_ok = close({key: summary[key] for key in recomputed}, recomputed)
    dependency = json.loads((root / "confirmation_dependency.json").read_text())
    dependency_ok = all((
        json.loads((confirmation_root / "summary.json").read_text())["gate"]["passed"],
        json.loads((confirmation_root / "execution_audit.json").read_text())["passed"],
        dependency["summary_sha256"] == sha(confirmation_root / "summary.json"),
        dependency["audit_sha256"] == sha(confirmation_root / "execution_audit.json"),
    ))
    terminal_ok = all((
        (root / "terminal_status.txt").read_text().strip() == "completed_qwen_native_writer_ledger_bridge",
        (root / "runner_exit_code.txt").read_text().strip() == "0",
    ))
    checks = {
        "terminal_status_and_exit": terminal_ok,
        "passing_confirmation_dependency": dependency_ok,
        "source_coverage_and_chunk_reconstruction": coverage and row_ok,
        "shard_hashes_model_and_resources": shard_ok,
        "independent_aggregates_bootstrap_and_gate": aggregate_ok,
        "confirmation_and_natural_benchmarks_unscored": (
            not summary["confirmation_scored"]
            and not summary["natural_benchmark_scored"]
        ),
    }
    result = {
        "protocol": "qwen_native_writer_ledger_bridge_audit_v1",
        "passed": all(checks.values()),
        "checks": checks,
        "hashes": hashes,
        "summary_sha256": sha(root / "summary.json"),
        "recomputed": recomputed,
    }
    write_json(output, result)
    if not result["passed"]:
        raise RuntimeError("native writer ledger bridge audit failed")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
