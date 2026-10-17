#!/usr/bin/env python3
"""Independently recompute the paired Level-B RULER matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
from typing import Any


CONTEXTS = (
    (4096, "4k"), (8192, "8k"), (16384, "16k"),
    (32768, "32k"), (65536, "64k"), (131072, "128k"),
    (262144, "256k"),
)
ARMS = ("no_ttt", "ttt")
TASK_PREFIXES = (
    "ruler_qa_squad", "ruler_qa_hotpotqa",
    "ruler_niah_single_1", "ruler_niah_single_2", "ruler_niah_single_3",
    "ruler_niah_multikey_1", "ruler_niah_multikey_2",
    "ruler_niah_multikey_3", "ruler_niah_multivalue",
    "ruler_niah_multiquery", "ruler_vt", "ruler_fwe", "ruler_cwe",
)


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluation_input_sha256(payload: dict[str, Any]) -> str:
    rows = []
    for key, value in payload.items():
        if not isinstance(value, dict) or "origin_prompt" not in value or "references" not in value:
            raise ValueError(f"prediction sample {key} lacks prompt/reference provenance")
        rows.append([key, value["origin_prompt"], value["references"]])
    canonical = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def audit(root: pathlib.Path) -> dict[str, Any]:
    errors: list[str] = []
    matrix_path = root / "matrix_summary.json"
    if not matrix_path.is_file():
        return {"protocol": "in_place_ttt_level_b_ruler_matrix_audit_v1",
                "passed": False, "errors": ["missing matrix_summary.json"]}
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    recomputed: dict[str, dict[str, float]] = {}
    single_bindings: dict[str, dict[str, dict[str, str]]] = {arm: {} for arm in ARMS}
    for context, suffix in CONTEXTS:
        arm_scores: dict[str, float] = {}
        arm_inputs: dict[str, dict[str, str]] = {}
        for arm in ARMS:
            run_root = root / arm / suffix
            audit_path = run_root / "independent_audit.json"
            metadata_path = run_root / "run_metadata.json"
            summary_path = run_root / "summary.json"
            result_dirs = sorted(
                path for path in (run_root / "opencompass").glob("*/results/*") if path.is_dir()
            )
            prediction_dirs = sorted(
                path for path in (run_root / "opencompass").glob("*/predictions/*") if path.is_dir()
            )
            if (not audit_path.is_file() or not metadata_path.is_file()
                    or not summary_path.is_file() or len(result_dirs) != 1
                    or len(prediction_dirs) != 1):
                errors.append(f"missing audited raw run: {arm}/{suffix}")
                continue
            single = json.loads(audit_path.read_text(encoding="utf-8"))
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (single.get("passed") is not True or single.get("arm") != arm
                    or single.get("context_length") != context
                    or metadata.get("arm") != arm or metadata.get("context_length") != context):
                errors.append(f"single audit/metadata identity differs: {arm}/{suffix}")
            expected_tasks = {f"{prefix}_{suffix}" for prefix in TASK_PREFIXES}
            result_files = sorted(result_dirs[0].glob("*.json"))
            prediction_files = sorted(prediction_dirs[0].glob("*.json"))
            if {path.stem for path in result_files} != expected_tasks:
                errors.append(f"raw result task set differs: {arm}/{suffix}")
            if {path.stem for path in prediction_files} != expected_tasks:
                errors.append(f"raw prediction task set differs: {arm}/{suffix}")
            result_hashes = {path.name: sha256(path) for path in result_files}
            prediction_hashes = {path.name: sha256(path) for path in prediction_files}
            bindings = single.get("bindings", {})
            if bindings.get("raw_result_sha256") != result_hashes:
                errors.append(f"single audit result hashes are stale: {arm}/{suffix}")
            if bindings.get("raw_prediction_sha256") != prediction_hashes:
                errors.append(f"single audit prediction hashes are stale: {arm}/{suffix}")
            input_hashes = {}
            for path in prediction_files:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict) or list(payload) != [str(i) for i in range(100)]:
                    errors.append(f"raw prediction sample set differs: {arm}/{suffix}/{path.name}")
                    continue
                try:
                    input_hashes[path.stem] = evaluation_input_sha256(payload)
                except ValueError as exc:
                    errors.append(f"{arm}/{suffix}: {exc}")
            arm_inputs[arm] = input_hashes
            scores = []
            for path in result_files:
                score = float(json.loads(path.read_text(encoding="utf-8"))["score"])
                if not math.isfinite(score):
                    errors.append(f"non-finite raw score: {arm}/{suffix}/{path.name}")
                scores.append(score)
            if len(scores) != 13:
                errors.append(f"raw task count differs: {arm}/{suffix}")
            else:
                arm_scores[arm] = sum(scores) / 13
            single_bindings[arm][suffix] = {
                "independent_audit_sha256": sha256(audit_path),
                "run_metadata_sha256": sha256(metadata_path),
                "summary_sha256": sha256(summary_path),
            }
            matrix_binding = matrix.get("bindings", {}).get(arm, {}).get(suffix, {})
            if (matrix_binding.get("independent_audit_sha256") != sha256(audit_path)
                    or matrix_binding.get("summary_sha256") != sha256(summary_path)):
                errors.append(f"matrix single-run binding differs: {arm}/{suffix}")
        if set(arm_scores) == set(ARMS):
            recomputed[suffix] = {
                "no_ttt": arm_scores["no_ttt"],
                "ttt": arm_scores["ttt"],
                "ttt_minus_no_ttt": arm_scores["ttt"] - arm_scores["no_ttt"],
            }
        if set(arm_inputs) == set(ARMS):
            normalized = {}
            for arm, hashes in arm_inputs.items():
                token = "_" + suffix
                normalized[arm] = {
                    name[:-len(token)]: value for name, value in hashes.items()
                    if name.endswith(token)
                }
            if normalized["no_ttt"] != normalized["ttt"]:
                errors.append(f"paired raw RULER inputs differ: {suffix}")

    observed = matrix.get("context_scores", {})
    if set(observed) != {suffix for _, suffix in CONTEXTS}:
        errors.append("matrix summary context set differs")
    for suffix, values in recomputed.items():
        for key, expected in values.items():
            try:
                actual = float(observed[suffix][key])
            except (KeyError, TypeError, ValueError):
                errors.append(f"matrix summary is missing {suffix}/{key}")
                continue
            if abs(actual - expected) >= 1e-12:
                errors.append(f"matrix summary differs at {suffix}/{key}")
    registered = [suffix for context, suffix in CONTEXTS if context <= 131072]
    if all(suffix in recomputed for suffix in registered):
        no_ttt_mean = sum(recomputed[suffix]["no_ttt"] for suffix in registered) / 6
        ttt_mean = sum(recomputed[suffix]["ttt"] for suffix in registered) / 6
        mean = matrix.get("registered_4k_to_128k_mean") or {}
        expected_mean = {
            "no_ttt": no_ttt_mean,
            "ttt": ttt_mean,
            "ttt_minus_no_ttt": ttt_mean - no_ttt_mean,
        }
        for key, expected in expected_mean.items():
            try:
                actual = float(mean[key])
            except (KeyError, TypeError, ValueError):
                errors.append(f"registered mean is missing {key}")
                continue
            if abs(actual - expected) >= 1e-12:
                errors.append(f"registered mean differs at {key}")
    if matrix.get("protocol") != "in_place_ttt_level_b_ruler_matrix_summary_v1" \
            or matrix.get("completed") is not True \
            or matrix.get("context_256k_is_extrapolation") is not True:
        errors.append("matrix summary protocol/completion/claim boundary differs")
    return {
        "protocol": "in_place_ttt_level_b_ruler_matrix_audit_v1",
        "passed": not errors,
        "errors": errors,
        "recomputed_context_scores": recomputed,
        "matrix_summary_sha256": sha256(matrix_path),
        "single_bindings": single_bindings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    result = audit(args.root.resolve())
    partial = output.with_name(output.name + ".partial")
    partial.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial.replace(output)
    print(json.dumps({"output": str(output), "passed": result["passed"],
                      "errors": result["errors"]}, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
