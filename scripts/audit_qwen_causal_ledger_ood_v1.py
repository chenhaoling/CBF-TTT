"""Independently audit the one-time causal-ledger lifecycle OOD evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

from scripts.audit_qwen_causal_ledger_development_v2 import expected_rows, read_jsonl
from tasks.build_qwen_causal_ledger_ood_v1 import OOD_SEED, PHASE, SUITES, load
from tasks.evaluate_qwen_causal_ledger_development_v2 import ORACLE, POLICIES, PRIMARY
from tasks.evaluate_qwen_causal_ledger_ood_v1 import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    EXPECTED_EPISODES,
    EXPECTED_ROWS,
    PROTOCOL,
    SCALE,
)
from tasks.pack_cbf_event_warmup import sha
from tasks.pilot_qwen_causal_ledger_residual import render_query
from tasks.train_cbf_event_writer import write_json


AUDIT_PROTOCOL = "qwen_causal_ledger_ood_v1_audit_v1"


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


def audit_bootstrap(primary, comparator):
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
    exact.sort(); nll.sort()
    bounds = lambda values: [
        values[max(0, math.ceil(0.025 * len(values)) - 1)],
        values[min(len(values) - 1, math.ceil(0.975 * len(values)) - 1)],
    ]
    return {
        "exact_gain": statistics.fmean(
            primary[group]["exact"] - comparator[group]["exact"] for group in groups
        ),
        "exact_bootstrap_95ci": bounds(exact),
        "nll_gain": statistics.fmean(
            comparator[group]["nll"] - primary[group]["nll"] for group in groups
        ),
        "nll_bootstrap_95ci": bounds(nll),
    }


def audit_aggregate(rows):
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
                ) for role in ("current", "historical", "anchor")
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
                } for suite in SUITES
            },
            "lengths": {
                str(length): statistics.fmean(
                    row["policies"][policy]["exact"] for row in rows
                    if row["length"] == length
                ) for length in lengths
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
    candidates = [name for name in POLICIES if name not in {PRIMARY, ORACLE}]
    strongest = max(candidates, key=lambda name: metrics[name]["exact"])
    comparison = audit_bootstrap(group_metrics[PRIMARY], group_metrics[strongest])
    oracle = statistics.fmean(
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
        "exact_oracle_agreement": oracle == 1.0,
    }
    return {
        "metrics": metrics,
        "strongest_non_oracle_comparator": strongest,
        "comparison": comparison,
        "oracle_agreement": oracle,
        "gate": {"checks": checks, "passed": all(checks.values())},
    }


def dependency_check(root, confirmation_root, development_root):
    root, confirmation, development = map(
        Path, (root, confirmation_root, development_root)
    )
    dependency = json.loads((root / "confirmation_dependency.json").read_text())
    confirmation_summary = json.loads((confirmation / "summary.json").read_text())
    confirmation_audit = json.loads((confirmation / "execution_audit.json").read_text())
    hashes = {
        "summary.json": sha(confirmation / "summary.json"),
        "execution_audit.json": sha(confirmation / "execution_audit.json"),
    }
    _, ood_pairs = load(root / "data")
    ood_ids = {inp["id"] for inp, _ in ood_pairs}
    earlier_ids = set()
    earlier_symbols = set()
    for path in (
        development / "data/development.inputs.jsonl",
        confirmation / "data/confirmation.inputs.jsonl",
    ):
        earlier_ids.update(
            json.loads(line)["id"] for line in path.read_text().splitlines() if line
        )
    for path in (
        development / "data/development.targets.jsonl",
        confirmation / "data/confirmation.targets.jsonl",
    ):
        for line in path.read_text().splitlines():
            if not line:
                continue
            target = json.loads(line)
            for events in target["semantic_events"]:
                for event in events:
                    if event["op"] == "assign":
                        earlier_symbols.update(event["facts"])
                        earlier_symbols.update(event["facts"].values())
    ood_symbols = set()
    for _, target in ood_pairs:
        for events in target["semantic_events"]:
            for event in events:
                if event["op"] == "assign":
                    ood_symbols.update(event["facts"])
                    ood_symbols.update(event["facts"].values())
    symbol_overlap = ood_symbols & earlier_symbols
    ok = all((
        confirmation_summary["gate"]["passed"],
        confirmation_audit["passed"],
        dependency["hashes"] == hashes,
        len(ood_ids) == EXPECTED_EPISODES,
        not (ood_ids & earlier_ids),
        not symbol_overlap,
    ))
    return ok, {
        "registered_hashes": dependency["hashes"],
        "ood_seed": OOD_SEED,
        "ood_episodes": len(ood_ids),
        "earlier_ood_id_overlap": len(ood_ids & earlier_ids),
        "earlier_ood_symbol_overlap": len(symbol_overlap),
    }


def run(args):
    from transformers import AutoTokenizer

    root, output = Path(args.root), Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    manifest, pairs = load(root / "data")
    expected, order = expected_rows(pairs, phase=PHASE)
    rows, resources, hashes = [], [], {}
    shard_ok = True
    for shard in (0, 1):
        row_path = root / f"ood_shard{shard}.jsonl"
        resource_path = root / f"ood_resource_shard{shard}.json"
        part = read_jsonl(row_path)
        resource = json.loads(resource_path.read_text())
        hashes[row_path.name], hashes[resource_path.name] = sha(row_path), sha(resource_path)
        shard_ok &= all(
            row["shard"] == shard and order[row["episode_id"]] % 2 == shard
            for row in part
        )
        shard_ok &= all((
            resource["protocol"] == PROTOCOL,
            resource["episodes"] == EXPECTED_EPISODES // 2,
            resource["rows"] == len(part) == EXPECTED_ROWS // 2,
            resource["rows_sha256"] == hashes[row_path.name],
            resource["data_manifest_sha256"] == sha(root / "data/manifest.json"),
            resource["scale"] == SCALE,
            not resource["loading_info"]["missing_keys"],
            not resource["loading_info"]["unexpected_keys"],
            not resource["loading_info"]["mismatched_keys"],
            resource["confirmation_generated"],
            resource["ood_generated"],
        ))
        rows.extend(part); resources.append(resource)
    coverage = all((
        len(rows) == EXPECTED_ROWS,
        len({(row["episode_id"], row["query_id"]) for row in rows}) == EXPECTED_ROWS,
        set(expected) == {(row["episode_id"], row["query_id"]) for row in rows},
    ))
    row_ok = True
    for row in rows:
        source = expected[row["episode_id"], row["query_id"]]
        answer_ids = tokenizer.encode(source["answer"]["answer"], add_special_tokens=False)
        prompt_hash = hashlib.sha256(render_query(source["query"]["text"]).encode()).hexdigest()
        row_ok &= all((
            row["protocol"] == PROTOCOL,
            row["group_id"] == source["inp"]["group_id"],
            row["family"] == source["target"]["family"],
            row["length"] == source["target"]["length"],
            row["block_index"] == source["block"],
            row["role"] == source["answer"]["role"],
            row["absent"] == source["answer"]["absent"],
            row["answer"] == source["answer"]["answer"],
            row["candidates"] == source["query"]["candidates"],
            row["visible_facts"] == source["visible_facts"],
            row["prompt_sha256"] == prompt_hash,
            set(row["policies"]) == set(POLICIES),
            len(answer_ids) == 5,
        ))
        for policy, value in row["policies"].items():
            expected_value = source["returned"][policy]
            row_ok &= all((
                value["returned_value"] == expected_value,
                value["candidate_prediction"] in source["query"]["candidates"],
                value["candidate_correct"]
                == int(value["candidate_prediction"] == source["answer"]["answer"]),
                len(value["generated_ids"]) == 5,
                value["exact"] == int(value["generated_ids"] == answer_ids),
                math.isfinite(value["nll"]),
            ))
    summary = json.loads((root / "summary.json").read_text())
    recomputed = audit_aggregate(rows)
    aggregate_ok = close(
        {key: summary[key] for key in recomputed}, recomputed
    )
    dependency_ok, dependency = dependency_check(
        root, args.confirmation_root, args.development_root
    )
    terminal_ok = all((
        (root / "terminal_status.txt").read_text().strip()
        == "completed_qwen_causal_ledger_ood_v1",
        (root / "runner_exit_code.txt").read_text().strip() == "0",
    ))
    sealed_ok = all((
        manifest["seed"] == OOD_SEED,
        manifest["confirmation_generated"],
        manifest["ood_generated"],
        summary["confirmation_generated"],
        summary["ood_generated"],
    ))
    checks = {
        "terminal_status_and_exit": terminal_ok,
        "passing_confirmation_dependency_and_fresh_ood": dependency_ok,
        "fresh_corpus_and_coverage": coverage,
        "shard_hashes_model_and_resources": shard_ok,
        "row_prompt_state_policy_and_generation": row_ok,
        "independent_aggregates_bootstrap_and_gate": aggregate_ok,
        "one_time_ood_markers": sealed_ok,
    }
    result = {
        "protocol": AUDIT_PROTOCOL,
        "passed": all(checks.values()),
        "checks": checks,
        "hashes": hashes,
        "data_manifest_sha256": sha(root / "data/manifest.json"),
        "summary_sha256": sha(root / "summary.json"),
        "confirmation_dependency": dependency,
        "recomputed": recomputed,
    }
    write_json(output, result)
    if not result["passed"]:
        raise RuntimeError("causal-ledger OOD audit failed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--confirmation-root", required=True)
    parser.add_argument("--development-root", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
