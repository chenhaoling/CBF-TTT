#!/usr/bin/env python3
"""Create a compact, deterministic summary of the RULER 4K smoke artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
import re


EXPECTED_TASKS = {
    "ruler_qa_squad_4k",
    "ruler_qa_hotpotqa_4k",
    "ruler_niah_single_1_4k",
    "ruler_niah_single_2_4k",
    "ruler_niah_single_3_4k",
    "ruler_niah_multikey_1_4k",
    "ruler_niah_multikey_2_4k",
    "ruler_niah_multikey_3_4k",
    "ruler_niah_multivalue_4k",
    "ruler_niah_multiquery_4k",
    "ruler_vt_4k",
    "ruler_fwe_4k",
    "ruler_cwe_4k",
}


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(root: pathlib.Path) -> dict:
    metadata_path = root / "run_metadata.json"
    log_path = root / "eval.log"
    exit_path = root / "exit_code.txt"
    metadata = json.loads(metadata_path.read_text())
    exit_code = int(exit_path.read_text().strip())

    result_dirs = sorted((root / "opencompass").glob("*/results/*"))
    if len(result_dirs) != 1:
        raise ValueError(f"expected one model result directory, found {result_dirs}")
    result_files = sorted(result_dirs[0].glob("*.json"))
    scores = {}
    result_hashes = {}
    for path in result_files:
        payload = json.loads(path.read_text())
        scores[path.stem] = float(payload["score"])
        result_hashes[path.name] = _sha256(path)

    observed = set(scores)
    complete_tasks = observed == EXPECTED_TASKS
    macro = sum(scores.values()) / len(scores) if scores else None

    peaks = {}
    with (root / "gpu.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            device = row["index"]
            peaks[device] = max(peaks.get(device, 0.0), float(row["memory_used_mib"]))

    log = log_path.read_text(errors="replace")
    elapsed_match = re.search(r"Elapsed \(wall clock\) time .*?:\s*([0-9:.]+)", log)
    summary_files = sorted((root / "opencompass").glob("*/summary/summary_*.txt"))
    official_text = summary_files[0].read_text(errors="replace") if len(summary_files) == 1 else ""
    official_match = re.search(
        r"ruler_4k: \{'score': ([0-9.]+), 'naive_average': ([0-9.]+)\}",
        official_text,
    )
    official_score = float(official_match.group(1)) if official_match else None

    summary = {
        "protocol": metadata["protocol"],
        "upstream_commit": metadata["upstream_commit"],
        "exit_code": exit_code,
        "completed": exit_code == 0 and complete_tasks,
        "context_length": metadata["context_length"],
        "num_samples_per_task": metadata["num_samples_per_task"],
        "task_count": len(scores),
        "missing_tasks": sorted(EXPECTED_TASKS - observed),
        "unexpected_tasks": sorted(observed - EXPECTED_TASKS),
        "task_scores": dict(sorted(scores.items())),
        "macro_accuracy": macro,
        "official_ruler_group_score": official_score,
        "macro_matches_official": (
            macro is not None
            and official_score is not None
            and abs(macro - official_score) < 1e-12
        ),
        "wall_clock": elapsed_match.group(1) if elapsed_match else None,
        "peak_gpu_memory_mib": peaks,
        "dependencies": {
            "plan_sha256": metadata["plan_sha256"],
            "config_sha256": metadata["config_sha256"],
            "model_config_sha256": metadata["model_config_sha256"],
            "model_hash_manifest_sha256": metadata.get(
                "model_hash_manifest_sha256"
            ),
            "eval_log_sha256": _sha256(log_path),
            "gpu_monitor_sha256": _sha256(root / "gpu.csv"),
            "official_summary_sha256": (
                _sha256(summary_files[0]) if len(summary_files) == 1 else None
            ),
            "result_sha256": result_hashes,
        },
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    payload = summarize(args.root)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
