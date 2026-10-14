"""Independently audit the dynamic-record resource pilot and excluded retry."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics

from tasks.build_qwen_dynamic_record_forgetting import FAMILIES, load_corpus
from tasks.pack_cbf_event_warmup import sha
from tasks.pilot_qwen_dynamic_record_forgetting import FORMAL_CHOICES, PROTOCOL
from tasks.train_cbf_event_writer import write_json


RESOURCE_FIELDS = {
    "label_seconds",
    "seconds_per_action",
    "peak_allocated_gib",
    "peak_reserved_gib",
}


def _rows(root, manifest):
    output = []
    for shard in manifest["shards"]:
        shard_root = root / shard["path"]
        if sha(shard_root / "manifest.json") != shard["manifest_sha256"]:
            raise ValueError("shard manifest hash mismatch")
        if sha(shard_root / "summary.json") != shard["summary_sha256"]:
            raise ValueError("shard summary hash mismatch")
        item = json.loads((shard_root / "manifest.json").read_text())
        for seed in item["projection_seeds"]:
            name = f"labels_seed{seed}.jsonl"
            path = shard_root / name
            if sha(path) != item["raw_rows_sha256"].get(name):
                raise ValueError("label row hash mismatch")
            output.extend(json.loads(line) for line in path.read_text().splitlines() if line)
    return sorted(output, key=lambda row: (
        row["projection_seed"], row["episode_id"], row["block_index"]
    ))


def _scientific_projection(rows):
    return [
        {key: value for key, value in row.items() if key not in RESOURCE_FIELDS}
        for row in rows
    ]


def _recompute(rows, shard_summaries):
    groups = len({row["group_id"] for row in rows})
    wall = max(row["wall_seconds"] for row in shard_summaries)
    extraction_allocated = max(
        row["extraction_peak_allocated_gib"] for row in shard_summaries
    )
    extraction_reserved = max(
        row["extraction_peak_reserved_gib"] for row in shard_summaries
    )
    label_allocated = max(row["peak_allocated_gib"] for row in rows)
    label_reserved = max(row["peak_reserved_gib"] for row in rows)
    overall_allocated = max(extraction_allocated, label_allocated)
    overall_reserved = max(extraction_reserved, label_reserved)
    estimates = {str(choice): wall * choice / groups for choice in FORMAL_CHOICES}
    eligible = [
        choice for choice in FORMAL_CHOICES
        if overall_allocated < 24.0 and estimates[str(choice)] <= 6 * 3600
    ]
    return {
        "rows": len(rows),
        "groups": groups,
        "projection_seeds": sorted({row["projection_seed"] for row in rows}),
        "families": {
            family: sum(row["family"] == family for row in rows)
            for family in FAMILIES
        },
        "wall_seconds": wall,
        "mean_label_seconds": statistics.fmean(row["label_seconds"] for row in rows),
        "p95_label_seconds": sorted(row["label_seconds"] for row in rows)[
            max(0, math.ceil(0.95 * len(rows)) - 1)
        ],
        "mean_seconds_per_action": statistics.fmean(
            row["seconds_per_action"] for row in rows
        ),
        "label_peak_allocated_gib": label_allocated,
        "label_peak_reserved_gib": label_reserved,
        "extraction_peak_allocated_gib": extraction_allocated,
        "extraction_peak_reserved_gib": extraction_reserved,
        "overall_peak_allocated_gib": overall_allocated,
        "overall_peak_reserved_gib": overall_reserved,
        "max_prompt_tokens": max(row["max_prompt_tokens"] for row in rows),
        "max_answer_tokens": max(row["max_answer_tokens"] for row in rows),
        "max_active_records": max(row["active_records"] for row in rows),
        "valid_action_counts": dict(Counter(str(row["valid_action_count"]) for row in rows)),
        "label_counts": dict(Counter(str(row["label"]) for row in rows)),
        "estimated_two_gpu_seconds": estimates,
        "selected_formal_train_groups": max(eligible) if eligible else None,
        "resource_gate_passed": bool(eligible),
    }


def run(args):
    root, excluded = Path(args.root), Path(args.excluded_root)
    manifest = json.loads((root / "manifest.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    excluded_manifest = json.loads((excluded / "manifest.json").read_text())
    if manifest.get("protocol") != PROTOCOL or excluded_manifest.get("protocol") != PROTOCOL:
        raise ValueError("unexpected dynamic resource protocol")
    _, pairs = load_corpus(root / "data")
    checks = {
        "runner_and_terminal_completed": (
            (root / "terminal_status.txt").read_text().strip()
            == "completed_qwen_dynamic_record_resource_pilot"
            and (root / "runner_exit_code.txt").read_text().strip() == "0"
        ),
        "registered_corpus_reloaded": len(pairs) == 112,
        "three_projection_shards_and_hashes": False,
        "excluded_attempt_completed_but_resource_invalid": False,
        "labels_reproduce_excluding_resource_fields": False,
        "row_structure_and_counts": False,
        "summary_recomputed": False,
        "resource_gate_recomputed": False,
        "formal_root_absent": not Path(args.formal_root).exists(),
    }
    rows = _rows(root, manifest)
    old_rows = _rows(excluded, excluded_manifest)
    checks["three_projection_shards_and_hashes"] = (
        {row["projection_seed"] for row in rows} == {2301, 2302, 2303}
    )
    old_summary = json.loads((excluded / "summary.json").read_text())
    checks["excluded_attempt_completed_but_resource_invalid"] = (
        (excluded / "terminal_status.txt").read_text().strip()
        == "completed_qwen_dynamic_record_resource_pilot"
        and "overall_peak_allocated_gib" not in old_summary
        and old_summary.get("peak_allocated_gib", 0.0) < 1.0
    )
    checks["labels_reproduce_excluding_resource_fields"] = (
        _scientific_projection(rows) == _scientific_projection(old_rows)
    )
    expected_rows = 8 * len(FAMILIES) * 2 * 3 * 3
    checks["row_structure_and_counts"] = (
        len(rows) == expected_rows
        and all(row["protocol"] == PROTOCOL for row in rows)
        and all(row["family"] in FAMILIES for row in rows)
        and all(row["valid_action_count"] in (3, 9, 27) for row in rows)
        and all(0 <= row["label_index"] < 54 for row in rows)
        and all(row["label_accuracy"] in (0.0, 1 / 3, 2 / 3, 1.0) for row in rows)
    )
    shard_summaries = [
        json.loads((root / shard["path"] / "summary.json").read_text())
        for shard in manifest["shards"]
    ]
    recomputed = _recompute(rows, shard_summaries)
    comparable = {
        key: summary[key] for key in recomputed
    }
    checks["summary_recomputed"] = comparable == recomputed
    checks["resource_gate_recomputed"] = (
        recomputed["resource_gate_passed"] is True
        and recomputed["selected_formal_train_groups"] == 512
        and recomputed["overall_peak_allocated_gib"] < 24.0
        and recomputed["estimated_two_gpu_seconds"]["512"] <= 6 * 3600
    )
    audit = {
        "protocol": "audit_qwen_dynamic_record_resource_pilot_v1",
        "audit_passed": all(checks.values()),
        "checks": checks,
        "summary": recomputed,
        "summary_sha256": sha(root / "summary.json"),
        "manifest_sha256": sha(root / "manifest.json"),
        "data_manifest_sha256": sha(root / "data" / "manifest.json"),
        "excluded_summary_sha256": sha(excluded / "summary.json"),
        "terminal_status": "completed_qwen_dynamic_record_resource_pilot_audit",
    }
    if not audit["audit_passed"]:
        raise RuntimeError(json.dumps(audit, indent=2))
    write_json(args.output, audit)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--excluded-root", required=True)
    parser.add_argument("--formal-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
