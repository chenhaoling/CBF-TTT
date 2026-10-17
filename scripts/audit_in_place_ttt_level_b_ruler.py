#!/usr/bin/env python3
"""Independently audit one formal Level-B RULER context run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pathlib
import re
from typing import Any


PROTOCOL = "in_place_ttt_level_b_ruler_audit_v1"
UPSTREAM_COMMIT = "be2324829b0e91c8fd10a74d4b43714fde6676e1"
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
EXPECTED_GENERATOR_SHA256 = {
    "opencompass.configs.datasets.ruler.ruler_qa_gen": "2c48ffb84f1ef858bd02ba333254defb67c528d3bd5142fe720c84591d4e1f06",
    "opencompass.configs.datasets.ruler.ruler_niah_gen": "e9c51a74a75428af7d1b6c2aafabbf1c93f817a1197e4b03301d66169837a77d",
    "opencompass.configs.datasets.ruler.ruler_vt_gen": "ce54adedba8ab4548809b4e3f0bce848f30acac87e6a1c2bdeffdb1de343aaa6",
    "opencompass.configs.datasets.ruler.ruler_fwe_gen": "9c54073a4c0fe687f0ed51e559ceb79041d28ab04258111017341111d8851c8d",
    "opencompass.configs.datasets.ruler.ruler_cwe_gen": "c8c0f87acc8f28b4ca6b3580223180ec06a30fdc140809d81fb66a1074efd24f",
    "opencompass.datasets.ruler.ruler_qa": "a753601befe8e6f4341b1c40ff4ebc710429e8f2098ead00e79aab8929a63da7",
    "opencompass.datasets.ruler.ruler_niah": "397b2b023c9c88fdc36041d82a71de21e7b71979893862173e628af832bcef7b",
    "opencompass.datasets.ruler.ruler_vt": "0501a09f88ec6bd919019a5be8516febee1bffe5229b7b174a1d35df2607c73a",
    "opencompass.datasets.ruler.ruler_fwe": "c900418620df5fb25363d9a0eda4fa5cf42ca4fb272772ee03ef006880ace194",
    "opencompass.datasets.ruler.ruler_cwe": "bf4a5b2e9168dc90030f420f9afda2a7f8c59cfe31306c46f55ee2dc4269edc3",
}
EXPECTED_SOURCE_SHA256 = {
    "data/ruler/dev-v2.0.json": "80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8",
    "data/ruler/hotpotqa.json": "4e9ecb5c8d3b719f624d66b60f8d56bf227f03914f5f0753d6fa1b359d7104ea",
    "data/ruler/PaulGrahamEssays.jsonl": "78ad877447201471578c69dd4fc11e6f489970966329468c5d0ac32292d53056",
}


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _one_dir(root: pathlib.Path, kind: str) -> pathlib.Path | None:
    found = sorted(path for path in (root / "opencompass").glob(f"*/{kind}/*") if path.is_dir())
    return found[0] if len(found) == 1 else None


def evaluation_input_sha256(payload: dict[str, Any]) -> str:
    rows = []
    for key, value in payload.items():
        if not isinstance(value, dict) or "origin_prompt" not in value or "references" not in value:
            raise ValueError(f"prediction sample {key} lacks prompt/reference provenance")
        rows.append([key, value["origin_prompt"], value["references"]])
    canonical = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def audit(root: pathlib.Path, plan: pathlib.Path, addendum: pathlib.Path) -> dict[str, Any]:
    errors: list[str] = []
    required = {name: root / name for name in (
        "run_metadata.json", "summary.json", "eval.log", "exit_code.txt", "gpu.csv",
    )}
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        return {"protocol": PROTOCOL, "passed": False,
                "errors": [f"missing required artifacts: {missing}"]}
    metadata = json.loads(required["run_metadata.json"].read_text(encoding="utf-8"))
    summary = json.loads(required["summary.json"].read_text(encoding="utf-8"))
    context = metadata.get("context_length")
    arm = metadata.get("arm")
    suffix = CONTEXT_SUFFIX.get(context)
    if metadata.get("protocol") != "in_place_ttt_level_b_ruler_run_v1":
        errors.append("run metadata protocol differs")
    if arm not in {"no_ttt", "ttt"} or suffix is None:
        errors.append("arm or context is not registered")
        suffix = "invalid"
    if metadata.get("num_samples_per_task") != 100:
        errors.append("sample count is not 100")
    if metadata.get("ruler_random_seed") != 42:
        errors.append("RULER random seed is not 42")
    if (metadata.get("context_suffix") != suffix
            or metadata.get("extrapolation") is not (context == 262144)):
        errors.append("context suffix or extrapolation flag differs")
    if (metadata.get("tokenizer_policy") != "qwen_original_tokenizer_mistral_fix_false_v1"
            or metadata.get("fix_mistral_regex") is not False):
        errors.append("tokenizer policy differs")
    if metadata.get("upstream_commit") != UPSTREAM_COMMIT or metadata.get("upstream_clean") is not True:
        errors.append("official checkout binding is invalid")
    if metadata.get("tool_clean") is not True:
        errors.append("tool checkout was not clean")
    if metadata.get("plan_sha256") != sha256(plan):
        errors.append("public-corpus plan hash differs")
    if metadata.get("evaluation_addendum_sha256") != sha256(addendum):
        errors.append("RULER addendum hash differs")
    tool_root = pathlib.Path(__file__).resolve().parents[1]
    expected_tool_artifacts = {
        "eval_config/baseline_reproduction/qwen3_4b_formal_ruler.py",
        "scripts/summarize_in_place_ttt_level_b_ruler.py",
        "scripts/audit_in_place_ttt_level_b_ruler.py",
    }
    recorded_tool_artifacts = metadata.get("tool_artifacts", {})
    if set(recorded_tool_artifacts) != expected_tool_artifacts:
        errors.append("tool artifact set differs")
    else:
        for relative, recorded in recorded_tool_artifacts.items():
            path = tool_root / relative
            if not path.is_file() or sha256(path) != recorded:
                errors.append(f"tool artifact hash differs: {relative}")
    config_path = tool_root / "eval_config/baseline_reproduction/qwen3_4b_formal_ruler.py"
    if metadata.get("config_sha256") != sha256(config_path):
        errors.append("formal RULER config hash differs")
    expected_versions = {
        "opencompass": "0.5.4",
        "transformers": "4.57.3",
        "datasets": "2.21.0",
        "torch": "2.8.0+cu128",
    }
    if metadata.get("dependency_versions") != expected_versions:
        errors.append("formal evaluation dependency versions differ")
    generator_files = metadata.get("ruler_generator_files", {})
    if not isinstance(generator_files, dict) or set(generator_files) != set(EXPECTED_GENERATOR_SHA256):
        errors.append("RULER generator source manifest differs")
    else:
        for module, record in generator_files.items():
            path = pathlib.Path(record.get("path", ""))
            if (not path.is_file() or record.get("sha256") != sha256(path)
                    or record.get("sha256") != EXPECTED_GENERATOR_SHA256[module]):
                errors.append(f"RULER generator source differs: {module}")
    source_files = metadata.get("ruler_source_files", {})
    if not isinstance(source_files, dict) or set(source_files) != set(EXPECTED_SOURCE_SHA256):
        errors.append("RULER source-data manifest differs")
    else:
        for relative, record in source_files.items():
            path = pathlib.Path(record.get("path", ""))
            if (not path.is_file() or record.get("sha256") != sha256(path)
                    or record.get("sha256") != EXPECTED_SOURCE_SHA256[relative]):
                errors.append(f"RULER source data differs: {relative}")

    training_path = pathlib.Path(metadata.get("training_audit_path", ""))
    if not training_path.is_file() or metadata.get("training_audit_sha256") != sha256(training_path):
        errors.append("Stage-2 training audit binding is invalid")
    else:
        training = json.loads(training_path.read_text(encoding="utf-8"))
        if (not training.get("passed") or training.get("stage") != "stage2"
                or training.get("arm") != arm):
            errors.append("Stage-2 training audit did not release this arm")
        try:
            expected_model = pathlib.Path(training["final_hf_path"]).resolve()
            if expected_model != pathlib.Path(metadata["model_path"]).resolve():
                errors.append("evaluated model differs from the audited Stage-2 HF path")
        except (KeyError, TypeError):
            errors.append("training audit has no final HF path")

    model_root = pathlib.Path(metadata.get("model_path", ""))
    model_config_path = model_root / "config.json"
    if not model_config_path.is_file() or metadata.get("model_config_sha256") != sha256(model_config_path):
        errors.append("model config hash differs")
    else:
        model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
        if bool(model_config.get("ttt_mode", False)) != (arm == "ttt"):
            errors.append("model ttt_mode differs from arm")
        if model_config.get("max_position_embeddings") != 131072:
            errors.append("model is not the registered Stage-2 128K checkpoint")
    model_files = metadata.get("model_files", {})
    if not isinstance(model_files, dict) or not model_files:
        errors.append("model file manifest is empty")
    else:
        safetensors = 0
        actual_model_files = {
            str(path.relative_to(model_root))
            for path in model_root.rglob("*") if path.is_file()
        }
        if set(model_files) != actual_model_files:
            errors.append("model artifact set differs")
        for relative, record in model_files.items():
            path = model_root / relative
            if not path.is_file() or path.stat().st_size != record.get("size_bytes") \
                    or sha256(path) != record.get("sha256"):
                errors.append(f"model artifact differs: {relative}")
            if relative.endswith(".safetensors"):
                safetensors += 1
        if safetensors == 0:
            errors.append("model manifest contains no safetensors")

    tokenizer_root = pathlib.Path(metadata.get("tokenizer_path", ""))
    for filename, key in (
        ("tokenizer.json", "tokenizer_json_sha256"),
        ("tokenizer_config.json", "tokenizer_config_sha256"),
    ):
        path = tokenizer_root / filename
        if not path.is_file() or metadata.get(key) != sha256(path):
            errors.append(f"tokenizer binding differs: {filename}")
    model_tokenizer = model_root / "tokenizer.json"
    if not model_tokenizer.is_file() or not (tokenizer_root / "tokenizer.json").is_file() \
            or sha256(model_tokenizer) != sha256(tokenizer_root / "tokenizer.json"):
        errors.append("model and original evaluation tokenizer differ")

    result_dir = _one_dir(root, "results")
    prediction_dir = _one_dir(root, "predictions")
    expected = {f"{prefix}_{suffix}" for prefix in TASK_PREFIXES}
    scores: dict[str, float] = {}
    if result_dir is None:
        errors.append("expected exactly one results model directory")
    else:
        files = sorted(result_dir.glob("*.json"))
        if {path.stem for path in files} != expected:
            errors.append("result task set differs")
        for path in files:
            payload = json.loads(path.read_text(encoding="utf-8"))
            score = float(payload.get("score", float("nan")))
            if not math.isfinite(score):
                errors.append(f"non-finite score: {path.name}")
            scores[path.stem] = score
            details = payload.get("details", {})
            if {key for key in details if str(key).isdigit()} != {str(i) for i in range(100)}:
                errors.append(f"result sample set differs: {path.name}")
    if prediction_dir is None:
        errors.append("expected exactly one predictions model directory")
    else:
        files = sorted(prediction_dir.glob("*.json"))
        if {path.stem for path in files} != expected:
            errors.append("prediction task set differs")
        for path in files:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or list(payload) != [str(i) for i in range(100)]:
                errors.append(f"prediction sample order differs: {path.name}")
            try:
                input_hash = evaluation_input_sha256(payload)
            except ValueError as exc:
                errors.append(str(exc))
                continue
            if summary.get("evaluation_input_sha256", {}).get(path.stem) != input_hash:
                errors.append(f"summary input hash differs: {path.name}")

    raw_result_hashes = {
        path.name: sha256(path) for path in sorted(result_dir.glob("*.json"))
    } if result_dir else {}
    raw_prediction_hashes = {
        path.name: sha256(path) for path in sorted(prediction_dir.glob("*.json"))
    } if prediction_dir else {}

    macro = sum(scores.values()) / len(scores) if scores else None
    if (summary.get("protocol") != "in_place_ttt_level_b_ruler_summary_v1"
            or summary.get("completed") is not True or summary.get("arm") != arm
            or summary.get("context_length") != context):
        errors.append("summary protocol/completion/identity differs")
    if macro is None or summary.get("macro_accuracy") is None \
            or abs(macro - float(summary["macro_accuracy"])) >= 1e-12:
        errors.append("summary macro does not reproduce raw task scores")
    if summary.get("macro_matches_official") is not True:
        errors.append("official RULER aggregate does not match raw macro")
    summary_bindings = summary.get("bindings", {})
    if summary_bindings.get("run_metadata_sha256") != sha256(required["run_metadata.json"]):
        errors.append("summary does not bind run metadata")
    if summary_bindings.get("result_sha256") != raw_result_hashes:
        errors.append("summary result hashes differ from raw artifacts")
    if summary_bindings.get("prediction_sha256") != raw_prediction_hashes:
        errors.append("summary prediction hashes differ from raw artifacts")
    official_files = sorted((root / "opencompass").glob("*/summary/summary_*.txt"))
    official_score = None
    if len(official_files) != 1:
        errors.append("expected exactly one official summary file")
    else:
        official_text = official_files[0].read_text(encoding="utf-8", errors="replace")
        match = re.search(
            rf"^ruler_{re.escape(suffix)}: \{{'score': ([-+0-9.eE]+),\s*"
            rf"'naive_average': ([-+0-9.eE]+)\}}$",
            official_text,
            flags=re.MULTILINE,
        )
        if match is None:
            errors.append("official RULER aggregate is missing")
        else:
            official_score = float(match.group(1))
            official_naive = float(match.group(2))
            if (macro is None or abs(macro - official_score) >= 1e-12
                    or abs(macro - official_naive) >= 1e-12):
                errors.append("official RULER aggregate differs from raw macro")
    if int(required["exit_code.txt"].read_text().strip()) != 0:
        errors.append("evaluation exit code is nonzero")
    log = required["eval.log"].read_text(encoding="utf-8", errors="replace")
    if "out of memory" in log.lower() or log.count("incorrect regex pattern") != 0:
        errors.append("evaluation log contains OOM or tokenizer regex warning")

    gpu_rows = 0
    peak_used = 0.0
    total_values: set[float] = set()
    with required["gpu.csv"].open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            gpu_rows += 1
            peak_used = max(peak_used, float(row["memory_used_mib"]))
            total_values.add(float(row["memory_total_mib"]))
            if int(row["index"]) != metadata.get("physical_gpu"):
                errors.append("GPU trace contains an unselected device")
                break
    if gpu_rows == 0 or len(total_values) != 1 or peak_used > next(iter(total_values), 0):
        errors.append("GPU trace is empty or invalid")

    return {
        "protocol": PROTOCOL,
        "passed": not errors,
        "errors": errors,
        "arm": arm,
        "context_length": context,
        "context_suffix": suffix,
        "extrapolation": context == 262144,
        "task_count": len(scores),
        "macro_accuracy": macro,
        "official_ruler_group_score": official_score,
        "gpu": {"rows": gpu_rows, "peak_memory_used_mib": peak_used,
                "memory_total_mib": next(iter(total_values), None)},
        "bindings": {
            **{name + "_sha256": sha256(path) for name, path in required.items()},
            "raw_result_sha256": raw_result_hashes,
            "raw_prediction_sha256": raw_prediction_hashes,
            "official_summary_sha256": (
                sha256(official_files[0]) if len(official_files) == 1 else None
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, required=True)
    parser.add_argument("--plan", type=pathlib.Path, required=True)
    parser.add_argument("--evaluation-addendum", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    result = audit(args.root.resolve(), args.plan.resolve(), args.evaluation_addendum.resolve())
    partial = output.with_name(output.name + ".partial")
    partial.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial.replace(output)
    print(json.dumps({"output": str(output), "passed": result["passed"],
                      "errors": result["errors"]}, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
