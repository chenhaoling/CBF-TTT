"""Independently audit causal-ledger development V2 and its release gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

from tasks.build_qwen_causal_ledger_development_v2 import load, PHASE
from tasks.build_qwen_causal_ledger_residual import FAMILIES, parse_visible_block
from tasks.evaluate_qwen_causal_ledger_development_v2 import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    ORACLE,
    POLICIES,
    PRIMARY,
    PROTOCOL,
    SCALE,
    append_record,
    lookup_value,
)
from tasks.pack_cbf_event_warmup import sha
from tasks.pilot_qwen_causal_ledger_residual import (
    assignment,
    render_query,
    semantic_apply,
    visible_facts,
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


def bootstrap(primary, comparator):
    groups = sorted(primary); rng = random.Random(BOOTSTRAP_SEED)
    exact, nll = [], []
    for _ in range(BOOTSTRAP_DRAWS):
        sample = [groups[rng.randrange(len(groups))] for _ in groups]
        exact.append(statistics.fmean(primary[group]["exact"] - comparator[group]["exact"] for group in sample))
        nll.append(statistics.fmean(comparator[group]["nll"] - primary[group]["nll"] for group in sample))
    exact.sort(); nll.sort()
    bounds = lambda values: [values[max(0, math.ceil(.025 * len(values)) - 1)], values[min(len(values) - 1, math.ceil(.975 * len(values)) - 1)]]
    return {
        "exact_gain": statistics.fmean(primary[group]["exact"] - comparator[group]["exact"] for group in groups),
        "exact_bootstrap_95ci": bounds(exact),
        "nll_gain": statistics.fmean(comparator[group]["nll"] - primary[group]["nll"] for group in groups),
        "nll_bootstrap_95ci": bounds(nll),
    }


def aggregate(rows):
    metrics, group_metrics = {}, {}
    groups = sorted({row["group_id"] for row in rows})
    for policy in POLICIES:
        metrics[policy] = {
            "candidate_accuracy": statistics.fmean(row["policies"][policy]["candidate_correct"] for row in rows),
            "exact": statistics.fmean(row["policies"][policy]["exact"] for row in rows),
            "nll": statistics.fmean(row["policies"][policy]["nll"] for row in rows),
            "roles": {role: statistics.fmean(row["policies"][policy]["exact"] for row in rows if row["role"] == role) for role in ("current", "historical", "anchor")},
            "families": {family: statistics.fmean(row["policies"][policy]["exact"] for row in rows if row["family"] == family) for family in FAMILIES},
            "lengths": {str(length): statistics.fmean(row["policies"][policy]["exact"] for row in rows if row["length"] == length) for length in (8, 12, 16)},
            "absent": statistics.fmean(row["policies"][policy]["exact"] for row in rows if row["absent"]),
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
    comparison = bootstrap(group_metrics[PRIMARY], group_metrics[strongest])
    oracle = statistics.fmean(
        row["policies"][PRIMARY]["candidate_prediction"] == row["policies"][ORACLE]["candidate_prediction"]
        and row["policies"][PRIMARY]["generated_ids"] == row["policies"][ORACLE]["generated_ids"]
        for row in rows
    )
    family_gains = {family: metrics[PRIMARY]["families"][family] - metrics[strongest]["families"][family] for family in FAMILIES}
    length_gains = {length: metrics[PRIMARY]["lengths"][length] - metrics[strongest]["lengths"][length] for length in metrics[PRIMARY]["lengths"]}
    primary = metrics[PRIMARY]
    checks = {
        "primary_candidate_at_least_point_95": primary["candidate_accuracy"] >= 0.95,
        "primary_exact_at_least_point_85": primary["exact"] >= 0.85,
        "current_anchor_at_least_point_95": min(primary["roles"]["current"], primary["roles"]["anchor"]) >= 0.95,
        "historical_at_least_point_85": primary["roles"]["historical"] >= 0.85,
        "absent_at_least_point_90": primary["absent"] >= 0.90,
        "every_family_at_least_point_85": min(primary["families"].values()) >= 0.85,
        "gain_at_least_point_05": comparison["exact_gain"] >= 0.05,
        "gain_ci_lower_above_point_02": comparison["exact_bootstrap_95ci"][0] > 0.02,
        "nll_gain_ci_lower_above_zero": comparison["nll_bootstrap_95ci"][0] > 0,
        "no_negative_family_or_length_gain": min(list(family_gains.values()) + list(length_gains.values())) >= 0,
        "exact_oracle_agreement": oracle == 1.0,
    }
    return {
        "metrics": metrics,
        "strongest_non_oracle_comparator": strongest,
        "comparison": comparison,
        "family_gains": family_gains,
        "length_gains": length_gains,
        "oracle_agreement": oracle,
        "gate": {"checks": checks, "passed": all(checks.values())},
    }


def expected_rows(pairs):
    output, order = {}, {}
    for index, (inp, target) in enumerate(pairs):
        order[inp["id"]] = index
        answers = {row["id"]: row for row in target["answers"]}
        queries = {}
        for query in inp["queries"]:
            queries.setdefault(query["after_block"], []).append(query)
        states = {name: [] for name in ("full_ledger", "retain_latest", "global_boundary_clear", "last1", "last2")}
        boundaries = set(inp["decision_boundaries"])
        for block in inp["blocks"]:
            event = assignment(parse_visible_block(block["text"], PHASE))
            states["full_ledger"] = semantic_apply(states["full_ledger"], event, block["index"])
            states["retain_latest"] = append_record(states["retain_latest"], event, block["index"])
            global_state = [] if block["index"] in boundaries else states["global_boundary_clear"]
            states["global_boundary_clear"] = append_record(global_state, event, block["index"])
            states["last1"] = append_record(states["last1"], event, block["index"])[-1:]
            states["last2"] = append_record(states["last2"], event, block["index"])[-2:]
            for query in queries.get(block["index"], []):
                answer = answers[query["id"]]
                returned = {
                    "no_memory": inp["null_value"],
                    "retain_latest": lookup_value(states["retain_latest"], query["key"], inp["null_value"]),
                    "global_boundary_clear": lookup_value(states["global_boundary_clear"], query["key"], inp["null_value"]),
                    "last1": lookup_value(states["last1"], query["key"], inp["null_value"]),
                    "last2": lookup_value(states["last2"], query["key"], inp["null_value"]),
                    "dense_full": None,
                    "full_ledger": lookup_value(states["full_ledger"], query["key"], inp["null_value"]),
                    "semantic_oracle": answer["answer"],
                }
                output[inp["id"], query["id"]] = {
                    "inp": inp, "target": target, "query": query, "answer": answer,
                    "block": block["index"], "returned": returned,
                    "visible_facts": len(visible_facts(states["full_ledger"])),
                }
    return output, order


def run(
    args,
    expected_protocol=PROTOCOL,
    expected_scale=SCALE,
    expected_terminal="completed_qwen_causal_ledger_development_v2",
    audit_protocol="qwen_causal_ledger_development_v2_audit_v1",
):
    from transformers import AutoTokenizer

    root, v2_root, output = Path(args.root), Path(args.v2_root), Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    manifest, pairs = load(root / "data")
    expected, order = expected_rows(pairs)
    rows, resources, hashes = [], [], {}
    shard_ok = True
    for shard in (0, 1):
        row_path = root / f"development_shard{shard}.jsonl"
        resource_path = root / f"development_resource_shard{shard}.json"
        part, resource = read_jsonl(row_path), json.loads(resource_path.read_text())
        hashes[row_path.name], hashes[resource_path.name] = sha(row_path), sha(resource_path)
        shard_ok &= all(row["shard"] == shard and order[row["episode_id"]] % 2 == shard for row in part)
        shard_ok &= all((
            resource["protocol"] == expected_protocol,
            resource["episodes"] == 448,
            resource["rows"] == len(part) == 4032,
            resource["rows_sha256"] == hashes[row_path.name],
            resource["data_manifest_sha256"] == sha(root / "data/manifest.json"),
            resource["scale"] == expected_scale,
            not resource["loading_info"]["missing_keys"],
            not resource["loading_info"]["unexpected_keys"],
            not resource["loading_info"]["mismatched_keys"],
            not resource["confirmation_generated"],
            not resource["ood_generated"],
        ))
        rows.extend(part); resources.append(resource)
    coverage = len(rows) == 8064 and len({(row["episode_id"], row["query_id"]) for row in rows}) == 8064 and set(expected) == {(row["episode_id"], row["query_id"]) for row in rows}
    row_ok = True
    for row in rows:
        source = expected[row["episode_id"], row["query_id"]]
        answer_ids = tokenizer.encode(source["answer"]["answer"], add_special_tokens=False)
        prompt_hash = hashlib.sha256(render_query(source["query"]["text"]).encode()).hexdigest()
        row_ok &= all((
            row["protocol"] == expected_protocol,
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
                value["candidate_correct"] == int(value["candidate_prediction"] == source["answer"]["answer"]),
                len(value["generated_ids"]) == 5,
                value["exact"] == int(value["generated_ids"] == answer_ids),
                math.isfinite(value["nll"]),
            ))
    summary = json.loads((root / "summary.json").read_text()); recomputed = aggregate(rows)
    aggregate_ok = close({key: summary[key] for key in recomputed}, recomputed)
    v2_summary, v2_audit = v2_root / "summary.json", v2_root / "execution_audit.json"
    dependency = json.loads((root / "v2_dependency.log").read_text())
    dependency_ok = all((
        json.loads(v2_summary.read_text())["gate"]["passed"],
        json.loads(v2_audit.read_text())["passed"],
        dependency["summary_sha256"] == sha(v2_summary),
        dependency["audit_sha256"] == sha(v2_audit),
    ))
    terminal = (root / "terminal_status.txt").read_text().strip() == expected_terminal and (root / "runner_exit_code.txt").read_text().strip() == "0"
    sealed = not summary["confirmation_generated"] and not summary["ood_generated"] and not manifest["confirmation_generated"] and not manifest["ood_generated"]
    checks = {
        "terminal_status_and_exit": terminal,
        "passing_v2_dependency_hashes": dependency_ok,
        "fresh_corpus_and_coverage": coverage,
        "shard_hashes_model_and_resources": shard_ok,
        "row_prompt_state_policy_and_generation": row_ok,
        "independent_aggregates_bootstrap_and_gate": aggregate_ok,
        "confirmation_and_ood_sealed": sealed,
    }
    result = {
        "protocol": audit_protocol,
        "passed": all(checks.values()),
        "checks": checks,
        "hashes": hashes,
        "data_manifest_sha256": sha(root / "data/manifest.json"),
        "summary_sha256": sha(root / "summary.json"),
        "recomputed": recomputed,
    }
    write_json(output, result)
    if not result["passed"]:
        raise RuntimeError("causal-ledger development V2 audit failed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--v2-root", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
