#!/usr/bin/env python3
"""Summarize one registered Level-B RULER context from raw OpenCompass output."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pathlib
import re
from typing import Any


CONTEXT_SUFFIX = {
    4096: "4k", 8192: "8k", 16384: "16k", 32768: "32k",
    65536: "64k", 131072: "128k", 262144: "256k",
}
TASK_PREFIXES = (
    "ruler_qa_squad", "ruler_qa_hotpotqa",
    "ruler_niah_single_1", "ruler_niah_single_2", "ruler_niah_single_3",
    "ruler_niah_multikey_1", "ruler_niah_multikey_2",
    "ruler_niah_multikey_3", "ruler_niah_multivalue",
    "ruler_niah_multiquery", "ruler_vt", "ruler_fwe", "ruler_cwe",
)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def expected_tasks(context: int) -> set[str]:
    suffix = CONTEXT_SUFFIX[context]
    return {f"{prefix}_{suffix}" for prefix in TASK_PREFIXES}


def evaluation_input_sha256(payload: dict[str, Any]) -> str:
    rows = []
    for key, value in payload.items():
        if not isinstance(value, dict) or "origin_prompt" not in value or "references" not in value:
            raise ValueError(f"prediction sample {key} lacks prompt/reference provenance")
        rows.append([key, value["origin_prompt"], value["references"]])
    canonical = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _one_model_dir(root: pathlib.Path, kind: str) -> pathlib.Path:
    found = sorted(path for path in (root / "opencompass").glob(f"*/{kind}/*") if path.is_dir())
    if len(found) != 1:
        raise ValueError(f"expected one OpenCompass {kind} model directory, found {found}")
    return found[0]


def summarize(root: pathlib.Path) -> dict[str, Any]:
    metadata_path = root / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    context = int(metadata["context_length"])
    if context not in CONTEXT_SUFFIX:
        raise ValueError(f"unregistered context {context}")
    expected = expected_tasks(context)
    result_dir = _one_model_dir(root, "results")
    prediction_dir = _one_model_dir(root, "predictions")
    result_files = sorted(result_dir.glob("*.json"))
    prediction_files = sorted(prediction_dir.glob("*.json"))

    scores: dict[str, float] = {}
    result_hashes: dict[str, str] = {}
    result_sample_counts: dict[str, int] = {}
    for path in result_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        score = float(payload["score"])
        if not math.isfinite(score):
            raise ValueError(f"non-finite score in {path}")
        scores[path.stem] = score
        details = payload.get("details", {})
        result_sample_counts[path.stem] = sum(str(key).isdigit() for key in details)
        result_hashes[path.name] = sha256(path)

    prediction_hashes: dict[str, str] = {}
    prediction_sample_counts: dict[str, int] = {}
    prediction_keys_complete: dict[str, bool] = {}
    input_hashes: dict[str, str] = {}
    expected_keys = [str(index) for index in range(100)]
    for path in prediction_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"prediction artifact is not a mapping: {path}")
        prediction_sample_counts[path.stem] = len(payload)
        prediction_keys_complete[path.stem] = list(payload) == expected_keys
        input_hashes[path.stem] = evaluation_input_sha256(payload)
        prediction_hashes[path.name] = sha256(path)

    observed_results = set(scores)
    observed_predictions = {path.stem for path in prediction_files}
    macro = sum(scores.values()) / len(scores) if scores else None
    log_path = root / "eval.log"
    log = log_path.read_text(encoding="utf-8", errors="replace")
    exit_code = int((root / "exit_code.txt").read_text(encoding="utf-8").strip())
    summary_files = sorted((root / "opencompass").glob("*/summary/summary_*.txt"))
    official_text = summary_files[0].read_text(encoding="utf-8", errors="replace") \
        if len(summary_files) == 1 else ""
    group_name = "ruler_" + CONTEXT_SUFFIX[context]
    official_match = re.search(
        rf"^{re.escape(group_name)}: \{{'score': ([-+0-9.eE]+),\s*"
        rf"'naive_average': ([-+0-9.eE]+)\}}$",
        official_text,
        flags=re.MULTILINE,
    )
    official_score = float(official_match.group(1)) if official_match else None
    official_naive = float(official_match.group(2)) if official_match else None

    peaks: dict[str, float] = {}
    with (root / "gpu.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            device = row["index"]
            peaks[device] = max(peaks.get(device, 0.0), float(row["memory_used_mib"]))

    elapsed_match = re.search(r"Elapsed \(wall clock\) time .*?:\s*([0-9:.]+)", log)
    exact_results = observed_results == expected
    exact_predictions = observed_predictions == expected
    hundred_each = (
        set(result_sample_counts) == expected
        and set(prediction_sample_counts) == expected
        and all(value == 100 for value in result_sample_counts.values())
        and all(value == 100 for value in prediction_sample_counts.values())
        and all(prediction_keys_complete.values())
    )
    macro_matches = (
        macro is not None and official_score is not None
        and abs(macro - official_score) < 1e-12
        and official_naive is not None and abs(macro - official_naive) < 1e-12
    )
    warning_count = log.count("incorrect regex pattern")
    completed = (
        exit_code == 0 and exact_results and exact_predictions and hundred_each
        and macro_matches and warning_count == 0 and len(summary_files) == 1
    )
    return {
        "protocol": "in_place_ttt_level_b_ruler_summary_v1",
        "completed": completed,
        "arm": metadata["arm"],
        "context_length": context,
        "context_suffix": CONTEXT_SUFFIX[context],
        "extrapolation": context == 262144,
        "num_samples_per_task": metadata["num_samples_per_task"],
        "task_count": len(scores),
        "missing_result_tasks": sorted(expected - observed_results),
        "unexpected_result_tasks": sorted(observed_results - expected),
        "missing_prediction_tasks": sorted(expected - observed_predictions),
        "unexpected_prediction_tasks": sorted(observed_predictions - expected),
        "result_sample_counts": dict(sorted(result_sample_counts.items())),
        "prediction_sample_counts": dict(sorted(prediction_sample_counts.items())),
        "prediction_keys_complete": dict(sorted(prediction_keys_complete.items())),
        "evaluation_input_sha256": dict(sorted(input_hashes.items())),
        "task_scores": dict(sorted(scores.items())),
        "macro_accuracy": macro,
        "official_ruler_group_score": official_score,
        "official_naive_average": official_naive,
        "macro_matches_official": macro_matches,
        "exit_code": exit_code,
        "wall_clock": elapsed_match.group(1) if elapsed_match else None,
        "tokenizer_regex_warning_count": warning_count,
        "peak_gpu_memory_mib": peaks,
        "bindings": {
            "run_metadata_sha256": sha256(metadata_path),
            "eval_log_sha256": sha256(log_path),
            "exit_code_sha256": sha256(root / "exit_code.txt"),
            "gpu_monitor_sha256": sha256(root / "gpu.csv"),
            "official_summary_sha256": sha256(summary_files[0]) if len(summary_files) == 1 else None,
            "result_sha256": result_hashes,
            "prediction_sha256": prediction_hashes,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    result = summarize(args.root.resolve())
    partial = output.with_name(output.name + ".partial")
    partial.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial.replace(output)
    print(json.dumps({"output": str(output), "completed": result["completed"]}, sort_keys=True))
    if not result["completed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
