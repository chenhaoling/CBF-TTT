"""Independently audit natural-value corpus, rows, gates and split release."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

from tasks.build_qwen_natural_value_bridge_v1 import (
    EXPECTED_COUNTS,
    EXPECTED_FILE_SHA256,
    PROTOCOL as DATA_PROTOCOL,
    SPLITS,
    build_split,
    load,
)
from tasks.evaluate_mquake_qwen_outcome_v16 import encode_candidates
from tasks.evaluate_qwen_natural_value_bridge_v1 import (
    ARMS,
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    PROTOCOL,
    QUERY_TYPES,
    SCALES,
    render_query,
)
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


AUDIT_PROTOCOL = "qwen_natural_value_bridge_v1_audit"


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
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    return left == right


def interval(values, seed):
    groups = sorted(values); rng = random.Random(seed); draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        sample = [groups[rng.randrange(len(groups))] for _ in groups]
        draws.append(statistics.fmean(values[group] for group in sample))
    draws.sort()
    return [
        draws[max(0, math.ceil(.025 * len(draws)) - 1)],
        draws[min(len(draws) - 1, math.ceil(.975 * len(draws)) - 1)],
    ]


def audit_aggregate(rows, scale):
    key = str(float(scale)); metrics = {}
    for arm in ARMS:
        metrics[arm] = {
            "candidate_accuracy": statistics.fmean(
                row["scales"][key]["arms"][arm]["candidate_correct"] for row in rows
            ),
            "answer_nll": statistics.fmean(
                row["scales"][key]["arms"][arm]["answer_nll"] for row in rows
            ),
            "by_query_type": {
                query_type: statistics.fmean(
                    row["scales"][key]["arms"][arm]["candidate_correct"]
                    for row in rows if row["query_type"] == query_type
                ) for query_type in QUERY_TYPES
            },
        }
    metrics["correct"]["generation_token_exact"] = statistics.fmean(
        row["scales"][key]["arms"]["correct"]["generation"]["token_exact"]
        for row in rows
    )
    metrics["correct"]["generation_by_query_type"] = {
        query_type: statistics.fmean(
            row["scales"][key]["arms"]["correct"]["generation"]["token_exact"]
            for row in rows if row["query_type"] == query_type
        ) for query_type in QUERY_TYPES
    }
    metrics["wrong"]["injected_wrong_accuracy"] = statistics.fmean(
        row["scales"][key]["arms"]["wrong"]["injected_wrong"] for row in rows
    )
    comparisons = {}; groups = sorted({row["episode_id"] for row in rows})
    for offset, control in enumerate(("wrong", "zero", "random")):
        values = {
            group: statistics.fmean(
                row["scales"][key]["arms"][control]["answer_nll"]
                - row["scales"][key]["arms"]["correct"]["answer_nll"]
                for row in rows if row["episode_id"] == group
            ) for group in groups
        }
        comparisons[control] = {
            "nll_gain": statistics.fmean(values.values()),
            "bootstrap_95ci": interval(values, BOOTSTRAP_SEED + offset),
        }
    strongest = max(("wrong", "zero", "random"), key=lambda arm: metrics[arm]["candidate_accuracy"])
    gains = {
        query_type: metrics["correct"]["by_query_type"][query_type]
        - metrics[strongest]["by_query_type"][query_type]
        for query_type in QUERY_TYPES
    }
    checks = {
        "correct_candidate_at_least_point_90": metrics["correct"]["candidate_accuracy"] >= .90,
        "correct_each_query_candidate_at_least_point_85": min(
            metrics["correct"]["by_query_type"].values()) >= .85,
        "correct_generation_at_least_point_70": metrics["correct"]["generation_token_exact"] >= .70,
        "correct_each_query_generation_at_least_point_65": min(
            metrics["correct"]["generation_by_query_type"].values()) >= .65,
        "wrong_truth_at_most_point_20": metrics["wrong"]["candidate_accuracy"] <= .20,
        "wrong_injected_at_least_point_70": metrics["wrong"]["injected_wrong_accuracy"] >= .70,
        "zero_truth_at_most_point_35": metrics["zero"]["candidate_accuracy"] <= .35,
        "all_nll_gain_ci_lower_above_zero": min(
            value["bootstrap_95ci"][0] for value in comparisons.values()) > 0,
    }
    return {
        "metrics": metrics,
        "comparisons": comparisons,
        "strongest_control": strongest,
        "query_type_candidate_gains": gains,
        "gate": {"checks": checks, "passed": all(checks.values())},
    }


def expected_data(root, source_root):
    root = Path(root); checks = []
    manifest = json.loads((root / "data/manifest.json").read_text())
    for split in SPLITS:
        expected = build_split(source_root, split)
        path = root / f"data/{split}.jsonl"
        content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in expected)
        checks.append(path.read_text() == content)
        checks.append(manifest["files_sha256"][path.name] == sha(path))
    checks.extend((
        manifest["protocol"] == DATA_PROTOCOL,
        not manifest["test_loaded"],
        not manifest["test_scored"],
        not manifest["external_benchmarks_loaded"],
    ))
    return all(checks), manifest


def read_and_check_stage(root, data_root, tokenizer, stage, scales):
    root = Path(root); _, episodes = load(data_root, stage)
    order = {episode["id"]: index for index, episode in enumerate(episodes)}
    expected = {(episode["id"], query_type): episode for episode in episodes for query_type in QUERY_TYPES}
    rows, resources = [], []
    shard_ok = True
    for shard in (0, 1):
        row_path = root / f"{stage}_shard{shard}.jsonl"
        resource_path = root / f"{stage}_resource_shard{shard}.json"
        part = [json.loads(line) for line in row_path.read_text().splitlines() if line]
        resource = json.loads(resource_path.read_text())
        shard_ok &= all(
            row["shard"] == shard and order[row["episode_id"]] % 2 == shard
            for row in part
        )
        shard_ok &= all((
            resource["protocol"] == PROTOCOL,
            resource["stage"] == stage,
            resource["episodes"] == EXPECTED_COUNTS[stage] // 2,
            resource["rows"] == len(part) == EXPECTED_COUNTS[stage],
            resource["rows_sha256"] == sha(row_path),
            resource["data_manifest_sha256"] == sha(Path(data_root) / "manifest.json"),
            resource["scales"] == list(scales),
            not resource["loading_info"]["missing_keys"],
            not resource["loading_info"]["unexpected_keys"],
            not resource["loading_info"]["mismatched_keys"],
            resource["calibration_scored"],
            resource["development_scored"] == (stage == "dev"),
            not resource["test_loaded"],
            not resource["test_scored"],
            not resource["external_benchmarks_loaded"],
        ))
        rows.extend(part); resources.append(resource)
    coverage = len(rows) == len(expected) and {
        (row["episode_id"], row["query_type"]) for row in rows
    } == set(expected)
    row_ok = True
    for row in rows:
        episode = expected[row["episode_id"], row["query_type"]]
        query = episode["queries"][row["query_type"]]
        prompt = render_query(query)
        prompt_ids, encoded = encode_candidates(tokenizer, prompt, episode["candidates"])
        row_ok &= all((
            row["protocol"] == PROTOCOL,
            row["split"] == stage,
            row["source_row_sha256"] == episode["source_row_sha256"],
            row["query_sha256"] == hashlib.sha256(query.encode()).hexdigest(),
            row["prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest(),
            row["correct"] == episode["correct"],
            row["wrong"] == episode["wrong"],
            row["candidates"] == episode["candidates"],
            row["candidate_token_ids"] == encoded,
            row["prompt_tokens"] == len(prompt_ids),
            row["correct_tokens"] == len(encoded[episode["correct"]]),
            row["wrong_tokens"] == len(encoded[episode["wrong"]]),
            set(map(float, row["scales"])) == set(scales),
            all(abs(value - 1.0) <= 1e-6 for value in row["random_row_norm_ratios"]),
            math.isfinite(row["seconds"]),
        ))
        for scale in scales:
            arm_rows = row["scales"][str(float(scale))]["arms"]
            row_ok &= set(arm_rows) == set(ARMS)
            for arm, value in arm_rows.items():
                prediction = min(episode["candidates"], key=lambda answer: (
                    value["scores"][answer], answer
                ))
                row_ok &= all((
                    set(value["scores"]) == set(episode["candidates"]),
                    all(math.isfinite(score) for score in value["scores"].values()),
                    value["prediction"] == prediction,
                    value["candidate_correct"] == int(prediction == episode["correct"]),
                    value["injected_wrong"] == int(prediction == episode["wrong"]),
                    close(value["answer_nll"], value["scores"][episode["correct"]]),
                ))
            generation = arm_rows["correct"]["generation"]
            row_ok &= all((
                generation["tokens"] == generation["conditioned_length"]
                == len(encoded[episode["correct"]]),
                generation["token_exact"]
                == int(generation["ids"] == encoded[episode["correct"]]),
            ))
    return rows, resources, shard_ok, coverage, row_ok


def dependency_check(root, ood_root, source_root):
    root, ood, source = map(Path, (root, ood_root, source_root))
    registered = json.loads((root / "dependencies.json").read_text())
    paths = (
        ood / "summary.json", ood / "execution_audit.json",
        source / "manifest.json", source / "calibration.jsonl", source / "dev.jsonl",
    )
    expected = {str(path): sha(path) for path in paths}
    ood_summary = json.loads((ood / "summary.json").read_text())
    ood_audit = json.loads((ood / "execution_audit.json").read_text())
    source_manifest = json.loads((source / "manifest.json").read_text())
    return all((
        registered == expected,
        ood_summary["gate"]["passed"],
        ood_audit["passed"],
        not source_manifest["test_loaded"],
        not source_manifest["test_scored"],
        sha(source / "calibration.jsonl") == EXPECTED_FILE_SHA256["calibration"],
        sha(source / "dev.jsonl") == EXPECTED_FILE_SHA256["dev"],
    )), expected


def run(args):
    from transformers import AutoTokenizer

    root, output = Path(args.root), Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    data_ok, data_manifest = expected_data(root, args.source_root)
    dependency_ok, dependencies = dependency_check(root, args.ood_root, args.source_root)
    calibration_rows, calibration_resources, calibration_shards, calibration_coverage, calibration_row_ok = read_and_check_stage(
        root, root / "data", tokenizer, "calibration", SCALES
    )
    calibration_summary = json.loads((root / "calibration_summary.json").read_text())
    calibration_tables = {
        str(scale): audit_aggregate(calibration_rows, scale) for scale in SCALES
    }
    passing = [scale for scale in SCALES if calibration_tables[str(scale)]["gate"]["passed"]]
    selected = min(passing) if passing else None
    calibration_recomputed = {
        "tables": calibration_tables,
        "selected_scale": selected,
        "gate": {
            "passed": selected is not None,
            "checks": {} if selected is None else calibration_tables[str(selected)]["gate"]["checks"],
        },
    }
    calibration_aggregate_ok = all((
        close(calibration_summary["tables"], calibration_tables),
        calibration_summary["selected_scale"] == selected,
        close(calibration_summary["gate"], calibration_recomputed["gate"]),
    ))
    development_present = selected is not None
    development_ok = True; development_recomputed = None
    if development_present:
        dev_rows, _, dev_shards, dev_coverage, dev_row_ok = read_and_check_stage(
            root, root / "data", tokenizer, "dev", (float(selected),)
        )
        development_recomputed = audit_aggregate(dev_rows, selected)
        development_recomputed["gate"]["checks"]["no_negative_query_type_candidate_gain"] = min(
            development_recomputed["query_type_candidate_gains"].values()
        ) >= 0
        development_recomputed["gate"]["passed"] = all(
            development_recomputed["gate"]["checks"].values()
        )
        development_summary = json.loads((root / "development_summary.json").read_text())
        keys = ("metrics", "comparisons", "strongest_control", "query_type_candidate_gains", "gate")
        development_ok = all((
            dev_shards, dev_coverage, dev_row_ok,
            close({key: development_summary[key] for key in keys}, development_recomputed),
            development_summary["selected_scale"] == selected,
            development_summary["development_scored"],
            not development_summary["test_loaded"],
            not development_summary["test_scored"],
        ))
    expected_terminal = (
        "completed_qwen_natural_value_bridge_v1"
        if development_present else "completed_qwen_natural_value_bridge_v1_calibration_stop"
    )
    terminal_ok = all((
        (root / "terminal_status.txt").read_text().strip() == expected_terminal,
        (root / "runner_exit_code.txt").read_text().strip() == "0",
    ))
    checks = {
        "terminal_status_and_exit": terminal_ok,
        "passing_ood_and_sealed_source_dependencies": dependency_ok,
        "model_free_corpus_reconstruction_and_hashes": data_ok,
        "calibration_shards_coverage_resources": calibration_shards and calibration_coverage,
        "calibration_rows_prompts_tokens_and_scores": calibration_row_ok,
        "independent_calibration_aggregates_scale_and_gate": calibration_aggregate_ok,
        "conditional_development_rows_and_gate": development_ok,
        "test_and_external_benchmarks_sealed": all((
            not data_manifest["test_loaded"], not data_manifest["test_scored"],
            not data_manifest["external_benchmarks_loaded"],
        )),
    }
    result = {
        "protocol": AUDIT_PROTOCOL,
        "passed": all(checks.values()),
        "checks": checks,
        "dependencies": dependencies,
        "data_manifest_sha256": sha(root / "data/manifest.json"),
        "calibration_summary_sha256": sha(root / "calibration_summary.json"),
        "development_summary_sha256": (
            sha(root / "development_summary.json") if development_present else None
        ),
        "calibration_recomputed": calibration_recomputed,
        "development_recomputed": development_recomputed,
    }
    write_json(output, result)
    if not result["passed"]:
        raise RuntimeError("natural-value bridge audit failed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--ood-root", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
