#!/usr/bin/env python3
"""Audit one one-sample 2x5090 RULER resource boundary."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

try:
    from scripts.audit_in_place_ttt_2x5090_ruler import (
        CONTEXT_SUFFIX, EXPECTED_GENERATOR_SHA256, EXPECTED_SOURCE_SHA256,
        TASK_PREFIXES, UPSTREAM_COMMIT, _one_dir, sha256,
    )
    from scripts.audit_in_place_ttt_paper_inference_overlay import (
        audit_paper_inference_overlay,
    )
except ModuleNotFoundError:  # Direct `python scripts/<name>.py` execution.
    from audit_in_place_ttt_2x5090_ruler import (
        CONTEXT_SUFFIX, EXPECTED_GENERATOR_SHA256, EXPECTED_SOURCE_SHA256,
        TASK_PREFIXES, UPSTREAM_COMMIT, _one_dir, sha256,
    )
    from audit_in_place_ttt_paper_inference_overlay import (
        audit_paper_inference_overlay,
    )


PROTOCOL = "in_place_ttt_2x5090_ruler_resource_audit_v1"


def audit(
    root: Path,
    corpus_plan: Path,
    evaluation_plan: Path,
    paper_inference_addendum: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    required = {name: root / name for name in (
        "run_metadata.json", "eval.log", "exit_code.txt", "gpu.csv",
        "gpu_before.csv", "preflight_host.txt", "preflight_conflicts.txt",
    )}
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        return {"protocol": PROTOCOL, "passed": False,
                "errors": [f"missing required artifacts: {missing}"]}
    metadata = json.loads(required["run_metadata.json"].read_text())
    arm = metadata.get("arm")
    context = metadata.get("context_length")
    suffix = CONTEXT_SUFFIX.get(context)
    if (metadata.get("protocol") != "in_place_ttt_2x5090_ruler_run_v1"
            or metadata.get("baseline_tier") != "2x5090_100m"
            or metadata.get("run_kind") != "resource"
            or metadata.get("num_samples_per_task") != 1
            or metadata.get("hostname") != "user-Super-Server"
            or metadata.get("physical_gpu") not in {0, 1}
            or arm not in {"no_ttt", "ttt"} or suffix is None):
        errors.append("resource metadata identity/count/host differs")
        suffix = suffix or "invalid"
    if (metadata.get("upstream_commit") != UPSTREAM_COMMIT
            or metadata.get("upstream_clean") is not True
            or metadata.get("tool_clean") is not True
            or not isinstance(metadata.get("tool_commit"), str)
            or len(metadata.get("tool_commit", "")) != 40):
        errors.append("repository binding is invalid")
    if metadata.get("public_corpus_plan_sha256") != sha256(corpus_plan):
        errors.append("public-corpus plan hash differs")
    if metadata.get("evaluation_plan_sha256") != sha256(evaluation_plan):
        errors.append("2x5090 RULER plan hash differs")

    tool_root = Path(__file__).resolve().parents[1]
    config = tool_root / "eval_config/baseline_reproduction/qwen3_4b_2x5090_ruler.py"
    expected_tools = {
        "eval_config/baseline_reproduction/qwen3_4b_2x5090_ruler.py",
        "scripts/run_in_place_ttt_2x5090_ruler_single.sh",
        "scripts/build_in_place_ttt_paper_inference_overlay.py",
        "scripts/validate_in_place_ttt_paper_inference_overlay.py",
        "scripts/audit_in_place_ttt_paper_inference_overlay.py",
        "scripts/audit_in_place_ttt_2x5090_ruler_resource.py",
    }
    recorded_tools = metadata.get("tool_artifacts", {})
    if set(recorded_tools) != expected_tools:
        errors.append("resource tool artifact set differs")
    else:
        for relative, digest in recorded_tools.items():
            path = tool_root / relative
            if not path.is_file() or sha256(path) != digest:
                errors.append(f"resource tool artifact differs: {relative}")
    errors.extend(audit_paper_inference_overlay(
        root, metadata, paper_inference_addendum, tool_root,
    ))
    if metadata.get("config_sha256") != sha256(config):
        errors.append("2x5090 RULER config differs")
    config_source = config.read_text(encoding="utf-8")
    if (config_source.count("paper_ttt_update_clip_norm = 1e-5") != 1
            or config_source.count(
                "model_kwargs=dict(ttt_update_clip_norm=paper_ttt_update_clip_norm)"
            ) != 1):
        errors.append("resource RULER config does not enable the paper update clip")

    for name, expected in (
        ("ruler_generator_files", EXPECTED_GENERATOR_SHA256),
        ("ruler_source_files", EXPECTED_SOURCE_SHA256),
    ):
        records = metadata.get(name, {})
        if not isinstance(records, dict) or set(records) != set(expected):
            errors.append(f"{name} set differs")
            continue
        for key, record in records.items():
            path = Path(record.get("path", ""))
            if (not path.is_file() or record.get("sha256") != sha256(path)
                    or record.get("sha256") != expected[key]):
                errors.append(f"{name} artifact differs: {key}")

    training_path = Path(metadata.get("training_audit_path", ""))
    arm_record: dict[str, Any] = {}
    if (not training_path.is_file()
            or metadata.get("training_audit_sha256") != sha256(training_path)):
        errors.append("formal Stage-2 audit binding is invalid")
    else:
        training = json.loads(training_path.read_text())
        if (training.get("protocol")
                != "in_place_ttt_2x5090_100m_stage2_audit_v1"
                or training.get("passed") is not True):
            errors.append("paired formal Stage-2 audit did not pass")
        try:
            arm_record = training["arms"][arm]
            expected_model = (
                training_path.parent / arm / arm_record["final_hf_relative_path"]
            ).resolve()
            if expected_model != Path(metadata["model_path"]).resolve():
                errors.append("resource model differs from formal Stage-2 HF path")
        except (KeyError, TypeError):
            errors.append("formal Stage-2 audit lacks the arm model")
    model = Path(metadata.get("model_path", ""))
    actual_model_files = {
        str(path.relative_to(model)): {
            "size_bytes": path.stat().st_size, "sha256": sha256(path),
        }
        for path in model.rglob("*") if path.is_file()
    } if model.is_dir() else {}
    if (not actual_model_files or metadata.get("model_files") != actual_model_files
            or arm_record.get("hf_manifest") != actual_model_files):
        errors.append("resource model manifest is empty, stale or unaudited")
    config_path = model / "config.json"
    if (not config_path.is_file()
            or metadata.get("model_config_sha256") != sha256(config_path)):
        errors.append("resource model config binding differs")
    else:
        config_data = json.loads(config_path.read_text())
        if (bool(config_data.get("ttt_mode", False)) != (arm == "ttt")
                or config_data.get("max_position_embeddings") != 131072):
            errors.append("resource model config differs from arm/Stage 2")
    tokenizer = Path(metadata.get("tokenizer_path", ""))
    for filename, key in (
        ("tokenizer.json", "tokenizer_json_sha256"),
        ("tokenizer_config.json", "tokenizer_config_sha256"),
    ):
        path = tokenizer / filename
        if not path.is_file() or metadata.get(key) != sha256(path):
            errors.append(f"resource tokenizer binding differs: {filename}")
    if (not (model / "tokenizer.json").is_file()
            or not (tokenizer / "tokenizer.json").is_file()
            or sha256(model / "tokenizer.json")
            != sha256(tokenizer / "tokenizer.json")):
        errors.append("resource model and evaluation tokenizer differ")

    expected_tasks = {f"{prefix}_{suffix}" for prefix in TASK_PREFIXES}
    result_dir = _one_dir(root, "results")
    prediction_dir = _one_dir(root, "predictions")
    scores: dict[str, float] = {}
    result_hashes: dict[str, str] = {}
    prediction_hashes: dict[str, str] = {}
    if result_dir is None or prediction_dir is None:
        errors.append("resource run lacks exactly one result/prediction directory")
    else:
        result_files = sorted(result_dir.glob("*.json"))
        prediction_files = sorted(prediction_dir.glob("*.json"))
        if {path.stem for path in result_files} != expected_tasks:
            errors.append("resource result task set differs")
        if {path.stem for path in prediction_files} != expected_tasks:
            errors.append("resource prediction task set differs")
        for path in result_files:
            payload = json.loads(path.read_text())
            score = float(payload.get("score", float("nan")))
            if not math.isfinite(score):
                errors.append(f"non-finite resource score: {path.name}")
            if {key for key in payload.get("details", {}) if str(key).isdigit()} != {"0"}:
                errors.append(f"resource result sample set differs: {path.name}")
            scores[path.stem] = score
            result_hashes[path.name] = sha256(path)
        for path in prediction_files:
            payload = json.loads(path.read_text())
            if not isinstance(payload, dict) or list(payload) != ["0"]:
                errors.append(f"resource prediction sample set differs: {path.name}")
            prediction_hashes[path.name] = sha256(path)

    try:
        exit_code = int(required["exit_code.txt"].read_text().strip())
    except ValueError:
        exit_code = -1
    log = required["eval.log"].read_text(encoding="utf-8", errors="replace")
    if exit_code != 0 or "out of memory" in log.lower() \
            or log.count("incorrect regex pattern"):
        errors.append("resource evaluation failed, OOMed or warned on tokenizer regex")

    with required["gpu_before.csv"].open(newline="") as handle:
        before = [row for row in csv.reader(handle) if row]
    before_valid = False
    if len(before) == 1:
        try:
            index, name, total, used, _ = (item.strip() for item in before[0])
            before_valid = (
                int(index) == metadata.get("physical_gpu")
                and "RTX 5090" in name and float(total) >= 32000
                and float(used) <= 128
            )
        except (ValueError, TypeError):
            before_valid = False
    match = re.search(
        r"^MemAvailable:\s+(\d+)\s+kB$",
        required["preflight_host.txt"].read_text(), re.MULTILINE,
    )
    if (not before_valid or match is None
            or int(match.group(1)) < 32 * 1024 * 1024
            or required["preflight_conflicts.txt"].read_text().strip()):
        errors.append("resource preflight GPU/host/process gate is invalid")

    rows = 0
    peak = 0.0
    totals: set[float] = set()
    with required["gpu.csv"].open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows += 1
            if int(row["index"]) != metadata.get("physical_gpu"):
                errors.append("resource GPU trace contains another device")
            peak = max(peak, float(row["memory_used_mib"]))
            totals.add(float(row["memory_total_mib"]))
    total = next(iter(totals), 0.0)
    if rows == 0 or len(totals) != 1 or total < 32000 or peak > 0.95 * total:
        errors.append("resource GPU trace violates the 5% headroom gate")

    return {
        "protocol": PROTOCOL,
        "passed": not errors,
        "errors": errors,
        "arm": arm,
        "context_length": context,
        "context_suffix": suffix,
        "task_count": len(scores),
        "scores_discarded": True,
        "gpu": {"rows": rows, "peak_memory_used_mib": peak,
                "memory_total_mib": total or None},
        "bindings": {
            **{name + "_sha256": sha256(path) for name, path in required.items()},
            "paper_inference_addendum_sha256": sha256(
                paper_inference_addendum
            ),
            "paper_inference_overlay_manifest_sha256": metadata.get(
                "paper_inference_overlay_manifest_sha256"
            ),
            "paper_inference_overlay_validation_sha256": metadata.get(
                "paper_inference_overlay_validation_sha256"
            ),
            "raw_result_sha256": result_hashes,
            "raw_prediction_sha256": prediction_hashes,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--public-corpus-plan", type=Path, required=True)
    parser.add_argument("--evaluation-plan", type=Path, required=True)
    parser.add_argument(
        "--paper-inference-addendum", type=Path, required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    result = audit(
        args.root.expanduser().resolve(),
        args.public_corpus_plan.expanduser().resolve(),
        args.evaluation_plan.expanduser().resolve(),
        args.paper_inference_addendum.expanduser().resolve(),
    )
    partial.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    partial.replace(output)
    print(json.dumps({"output": str(output), "passed": result["passed"],
                      "errors": result["errors"]}, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
