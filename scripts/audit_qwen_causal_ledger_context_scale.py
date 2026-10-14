"""Independently audit the causal-ledger contextual residual-scale diagnostic."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics

from tasks.build_qwen_causal_ledger_residual import load_pilot
from tasks.diagnose_qwen_causal_ledger_context_scale import PROTOCOL, SCALES, select_scale
from tasks.pack_cbf_event_warmup import sha
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


def aggregate(rows):
    table = {}
    for scale in SCALES:
        key = str(scale)
        table[key] = {
            "free_exact": statistics.fmean(row["by_scale"][key]["free_exact"] for row in rows),
            "free_text_exact": statistics.fmean(row["by_scale"][key]["free_text_exact"] for row in rows),
            "free_token_accuracy": statistics.fmean(row["by_scale"][key]["free_token_accuracy"] for row in rows),
            "free_position_accuracy": [
                statistics.fmean(row["by_scale"][key]["free_position_correct"][index] for row in rows)
                for index in range(5)
            ],
            "teacher_sequence_exact": statistics.fmean(row["by_scale"][key]["teacher_sequence_exact"] for row in rows),
            "teacher_position_accuracy": [
                statistics.fmean(row["by_scale"][key]["teacher_position_correct"][index] for row in rows)
                for index in range(5)
            ],
            "answer_nll": statistics.fmean(row["by_scale"][key]["answer_nll"] for row in rows),
            "family_free_exact": {
                family: statistics.fmean(
                    row["by_scale"][key]["free_exact"] for row in rows if row["family"] == family
                ) for family in sorted({row["family"] for row in rows})
            },
        }
    return table


def run(args):
    from transformers import AutoTokenizer

    root, source_root, output = Path(args.root), Path(args.source_root), Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    manifest, pairs = load_pilot(source_root / "data")
    order, truth = {}, {}
    for index, (inp, target) in enumerate(pairs):
        order[inp["id"]] = index
        answers = {row["id"]: row for row in target["answers"]}
        queries = {row["id"]: row for row in inp["queries"]}
        truth[inp["id"]] = (inp, target, answers, queries)

    rows, resources, hashes = [], [], {}
    shard_integrity = True
    for shard in (0, 1):
        row_path = root / f"scale_shard{shard}.jsonl"
        resource_path = root / f"scale_resource_shard{shard}.json"
        part, resource = read_jsonl(row_path), json.loads(resource_path.read_text())
        hashes[row_path.name], hashes[resource_path.name] = sha(row_path), sha(resource_path)
        shard_integrity &= all(
            row["shard"] == shard and order[row["episode_id"]] % 2 == shard for row in part
        )
        shard_integrity &= all((
            resource["protocol"] == PROTOCOL,
            resource["shard"] == shard,
            resource["num_shards"] == 2,
            resource["episodes"] == 56,
            resource["rows"] == len(part) == 504,
            resource["rows_sha256"] == hashes[row_path.name],
            resource["data_manifest_sha256"] == sha(source_root / "data/manifest.json"),
            resource["scales"] == list(SCALES),
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
        expected_ids = tokenizer.encode(answer["answer"], add_special_tokens=False)
        row_integrity &= all((
            row["protocol"] == PROTOCOL,
            row["group_id"] == inp["group_id"],
            row["family"] == target["family"],
            row["role"] == query["role"] == answer["role"],
            row["answer"] == answer["answer"],
            row["target_ids"] == expected_ids,
            len(expected_ids) == 5,
            len(row["residual_norms"]) == len(row["hidden_norms"]) == 5,
            all(value > 0 and math.isfinite(value) for value in row["residual_norms"] + row["hidden_norms"]),
            set(row["by_scale"]) == {str(scale) for scale in SCALES},
        ))
        for scale in SCALES:
            item = row["by_scale"][str(scale)]
            teacher_correct = [int(left == right) for left, right in zip(item["teacher_ids"], expected_ids)]
            free_correct = [int(left == right) for left, right in zip(item["generated_ids"], expected_ids)]
            decoded = tokenizer.decode(item["generated_ids"], skip_special_tokens=True).strip()
            normalized = lambda value: " ".join(value.strip().casefold().split())
            row_integrity &= all((
                len(item["teacher_ids"]) == len(item["generated_ids"]) == 5,
                item["teacher_position_correct"] == teacher_correct,
                item["teacher_sequence_exact"] == int(all(teacher_correct)),
                item["free_position_correct"] == free_correct,
                close(item["free_token_accuracy"], statistics.fmean(free_correct)),
                item["free_exact"] == int(item["generated_ids"] == expected_ids),
                item["generated_text"] == decoded,
                item["free_text_exact"] == int(normalized(decoded) == normalized(answer["answer"])),
                math.isfinite(item["answer_nll"]) and item["answer_nll"] >= 0,
            ))

    summary = json.loads((root / "summary.json").read_text())
    table = aggregate(rows); selected = select_scale(table)
    aggregate_integrity = all((
        close(summary["table"], table),
        summary["selected_scale"] == selected,
        summary["passed"] == (selected is not None),
        close(summary["resources"]["two_gpu_critical_path_seconds"], max(row["seconds"] for row in resources)),
        close(summary["resources"]["peak_allocated_gib"], max(row["peak_allocated_gib"] for row in resources)),
        close(summary["resources"]["peak_reserved_gib"], max(row["peak_reserved_gib"] for row in resources)),
    ))
    terminal = (
        (root / "terminal_status.txt").read_text().strip() == "completed_qwen_causal_ledger_context_scale"
        and (root / "runner_exit_code.txt").read_text().strip() == "0"
    )
    sealed = all(
        not summary[name] and not manifest[name]
        for name in ("development_generated", "confirmation_generated", "ood_generated")
    )
    checks = {
        "terminal_status_and_exit": terminal,
        "source_and_merged_coverage": coverage,
        "shard_hashes_and_resources": shard_integrity,
        "row_level_token_recomputation": row_integrity,
        "independent_aggregates_and_selection": aggregate_integrity,
        "downstream_splits_sealed": sealed,
    }
    result = {
        "protocol": "qwen_causal_ledger_context_scale_audit_v1",
        "passed": all(checks.values()),
        "checks": checks,
        "hashes": hashes,
        "source_manifest_sha256": sha(source_root / "data/manifest.json"),
        "summary_sha256": sha(root / "summary.json"),
        "recomputed_selected_scale": selected,
        "pilot_seed": manifest["seed"],
    }
    write_json(output, result)
    if not result["passed"]:
        raise RuntimeError("context-scale diagnostic audit failed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
