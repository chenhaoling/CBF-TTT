#!/usr/bin/env python3
"""Summarize the paired six-context 2x5090 100M-tier RULER matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from typing import Any


CONTEXTS = (
    (4096, "4k"), (8192, "8k"), (16384, "16k"),
    (32768, "32k"), (65536, "64k"), (131072, "128k"),
)
ARMS = ("no_ttt", "ttt")


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def summarize(root: pathlib.Path) -> dict[str, Any]:
    errors: list[str] = []
    runs: dict[str, dict[str, Any]] = {arm: {} for arm in ARMS}
    bindings: dict[str, dict[str, dict[str, str]]] = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for context, suffix in CONTEXTS:
            run_root = root / arm / suffix
            summary_path = run_root / "summary.json"
            audit_path = run_root / "independent_audit.json"
            if not summary_path.is_file() or not audit_path.is_file():
                errors.append(f"missing single-run summary/audit: {arm}/{suffix}")
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            if (summary.get("completed") is not True or audit.get("passed") is not True
                    or summary.get("arm") != arm or audit.get("arm") != arm
                    or summary.get("context_length") != context
                    or audit.get("context_length") != context):
                errors.append(f"single-run identity/completion differs: {arm}/{suffix}")
            runs[arm][suffix] = summary
            bindings[arm][suffix] = {
                "summary_sha256": sha256(summary_path),
                "independent_audit_sha256": sha256(audit_path),
            }

    context_scores: dict[str, dict[str, float]] = {}
    task_deltas: dict[str, dict[str, float]] = {}
    for _, suffix in CONTEXTS:
        if suffix not in runs["no_ttt"] or suffix not in runs["ttt"]:
            continue
        base = runs["no_ttt"][suffix]
        ttt = runs["ttt"][suffix]
        base_macro = float(base["macro_accuracy"])
        ttt_macro = float(ttt["macro_accuracy"])
        context_scores[suffix] = {
            "no_ttt": base_macro,
            "ttt": ttt_macro,
            "ttt_minus_no_ttt": ttt_macro - base_macro,
        }
        suffix_token = "_" + suffix
        base_scores = {
            name[:-len(suffix_token)]: float(value)
            for name, value in base["task_scores"].items()
            if name.endswith(suffix_token)
        }
        ttt_scores = {
            name[:-len(suffix_token)]: float(value)
            for name, value in ttt["task_scores"].items()
            if name.endswith(suffix_token)
        }
        if set(base_scores) != set(ttt_scores) or len(base_scores) != 13:
            errors.append(f"paired task set differs: {suffix}")
        base_inputs = base.get("evaluation_input_sha256")
        ttt_inputs = ttt.get("evaluation_input_sha256")
        if (not isinstance(base_inputs, dict) or len(base_inputs) != 13
                or not isinstance(ttt_inputs, dict) or len(ttt_inputs) != 13
                or base_inputs != ttt_inputs):
            errors.append(f"paired RULER inputs differ: {suffix}")
        task_deltas[suffix] = {
            task: ttt_scores[task] - base_scores[task]
            for task in sorted(set(base_scores) & set(ttt_scores))
        }

    registered_suffixes = [suffix for context, suffix in CONTEXTS if context <= 131072]
    have_registered = all(suffix in context_scores for suffix in registered_suffixes)
    registered_means = None
    if have_registered:
        no_ttt_mean = sum(context_scores[suffix]["no_ttt"] for suffix in registered_suffixes) / 6
        ttt_mean = sum(context_scores[suffix]["ttt"] for suffix in registered_suffixes) / 6
        registered_means = {
            "contexts": registered_suffixes,
            "no_ttt": no_ttt_mean,
            "ttt": ttt_mean,
            "ttt_minus_no_ttt": ttt_mean - no_ttt_mean,
        }
    complete = not errors and len(context_scores) == 6 and registered_means is not None
    return {
        "protocol": "in_place_ttt_2x5090_ruler_matrix_summary_v1",
        "baseline_tier": "2x5090_100m",
        "completed": complete,
        "errors": errors,
        "context_scores": context_scores,
        "task_deltas": task_deltas,
        "registered_4k_to_128k_mean": registered_means,
        "registered_contexts": registered_suffixes,
        "context_256k_status": "separate_extrapolation_resource_gate",
        "bindings": bindings,
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
    print(json.dumps({"output": str(output), "completed": result["completed"],
                      "errors": result["errors"]}, sort_keys=True))
    if not result["completed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
