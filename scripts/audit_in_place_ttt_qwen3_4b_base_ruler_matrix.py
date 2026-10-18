#!/usr/bin/env python3
"""Bind the formal RULER matrix to corrected Base prerequisite audits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


PROTOCOL = "in_place_ttt_qwen3_4b_base_ruler_matrix_audit_v1"
MODEL_REVISION = "906bfd4b4dc7f14ee4320094d8b41684abff8539"
CONTEXTS = {"4k", "8k", "16k", "32k", "64k", "128k"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def audit(args: argparse.Namespace) -> dict[str, Any]:
    underlying_path = args.root / "independent_matrix_audit.json"
    summary_path = args.root / "matrix_summary.json"
    metadata_path = args.root / "matrix_metadata.json"
    training_path = args.training_root / "independent_audit.json"
    resource_path = args.resource_root / "independent_resource_ladder_audit.json"
    required = [
        underlying_path, summary_path, metadata_path, training_path,
        args.corrected_training_audit, resource_path,
        args.corrected_resource_audit, args.model_identity,
        args.tokenizer / "tokenizer.json",
        args.tokenizer / "tokenizer_config.json", args.public_corpus_plan,
        args.evaluation_plan, args.corrected_plan,
        args.paper_inference_addendum, args.runner, args.underlying_auditor,
        args.summarizer,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return {"protocol": PROTOCOL, "passed": False,
                "errors": [f"missing required files: {missing}"]}
    errors: list[str] = []
    underlying, summary, metadata = (
        load(underlying_path), load(summary_path), load(metadata_path)
    )
    training, corrected_training = load(training_path), load(
        args.corrected_training_audit
    )
    resource, corrected_resource = load(resource_path), load(
        args.corrected_resource_audit
    )
    identity = load(args.model_identity)
    identity_files = identity.get("files", {})
    if (identity.get("passed") is not True
            or not identity.get("checks")
            or not all(identity.get("checks", {}).values())
            or identity.get("revision") != MODEL_REVISION):
        errors.append("Base identity audit is invalid")
    for name in ("tokenizer.json", "tokenizer_config.json"):
        path, saved = args.tokenizer / name, identity_files.get(name, {})
        if (path.stat().st_size != saved.get("size")
                or sha256(path) != saved.get("sha256")):
            errors.append(f"Base tokenizer asset drift: {name}")
    if (training.get("protocol")
            != "in_place_ttt_2x5090_100m_stage2_audit_v1"
            or training.get("passed") is not True or training.get("errors") != []
            or corrected_training.get("protocol")
            != "in_place_ttt_qwen3_4b_base_100m_stage2_audit_v1"
            or corrected_training.get("passed") is not True
            or corrected_training.get("errors") != []
            or corrected_training.get("model_revision") != MODEL_REVISION
            or corrected_training.get("underlying_formal_audit_sha256")
            != sha256(training_path)):
        errors.append("corrected formal Stage-2 prerequisite is invalid")
    if (resource.get("protocol")
            != "in_place_ttt_2x5090_ruler_resource_ladder_audit_v1"
            or resource.get("passed") is not True or resource.get("errors") != []
            or corrected_resource.get("protocol")
            != "in_place_ttt_qwen3_4b_base_ruler_resource_audit_v1"
            or corrected_resource.get("passed") is not True
            or corrected_resource.get("errors") != []
            or corrected_resource.get("model_revision") != MODEL_REVISION
            or corrected_resource.get("underlying_resource_audit_sha256")
            != sha256(resource_path)
            or corrected_resource.get("corrected_training_audit_sha256")
            != sha256(args.corrected_training_audit)):
        errors.append("corrected RULER resource prerequisite is invalid")
    if (underlying.get("protocol")
            != "in_place_ttt_2x5090_ruler_matrix_audit_v1"
            or underlying.get("passed") is not True
            or underlying.get("errors") != []
            or set(underlying.get("recomputed_context_scores", {})) != CONTEXTS):
        errors.append("underlying formal RULER matrix audit failed")
    scores = underlying.get("recomputed_context_scores", {})
    for suffix, record in scores.items():
        if (set(record) != {"no_ttt", "ttt", "ttt_minus_no_ttt"}
                or not all(isinstance(value, (int, float))
                           and not isinstance(value, bool)
                           and math.isfinite(value)
                           for value in record.values())):
            errors.append(f"{suffix}: corrected RULER score record is invalid")
    if (summary.get("protocol")
            != "in_place_ttt_2x5090_ruler_matrix_summary_v1"
            or summary.get("completed") is not True
            or summary.get("claim_boundary")
            != "public_data_matched_replication_not_paper_exact"
            or set(summary.get("context_scores", {})) != CONTEXTS):
        errors.append("formal RULER matrix summary is invalid")
    expected = {
        "training_audit_sha256": sha256(training_path),
        "resource_audit_sha256": sha256(resource_path),
        "public_corpus_plan_sha256": sha256(args.public_corpus_plan),
        "evaluation_plan_sha256": sha256(args.evaluation_plan),
        "paper_inference_addendum_sha256": sha256(
            args.paper_inference_addendum
        ),
        "runner_sha256": sha256(args.runner),
        "summarizer_sha256": sha256(args.summarizer),
        "auditor_sha256": sha256(args.underlying_auditor),
    }
    if (metadata.get("protocol")
            != "in_place_ttt_2x5090_ruler_matrix_run_v1"
            or metadata.get("hostname") != "user-Super-Server"
            or metadata.get("training_root")
            != str(args.training_root.resolve())
            or metadata.get("resource_root") != str(args.resource_root.resolve())
            or metadata.get("contexts")
            != [4096, 8192, 16384, 32768, 65536, 131072]
            or any(metadata.get(key) != value for key, value in expected.items())):
        errors.append("corrected RULER matrix metadata binding mismatch")
    dependency_bindings = underlying.get("dependency_bindings", {})
    if any(dependency_bindings.get(key) != value
           for key, value in expected.items()):
        errors.append("underlying RULER matrix binding is stale")
    return {
        "protocol": PROTOCOL,
        "passed": not errors,
        "errors": errors,
        "model_revision": MODEL_REVISION,
        "underlying_matrix_audit_sha256": sha256(underlying_path),
        "matrix_summary_sha256": sha256(summary_path),
        "corrected_training_audit_sha256": sha256(
            args.corrected_training_audit
        ),
        "corrected_resource_audit_sha256": sha256(
            args.corrected_resource_audit
        ),
        "context_scores": scores,
        "bindings": expected | {
            "corrected_plan_sha256": sha256(args.corrected_plan),
            "model_identity_sha256": sha256(args.model_identity),
        },
        "claim_boundary": "public_data_matched_replication_not_paper_exact",
        "update_admission_released": not errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--corrected-training-audit", type=Path, required=True)
    parser.add_argument("--resource-root", type=Path, required=True)
    parser.add_argument("--corrected-resource-audit", type=Path, required=True)
    parser.add_argument("--model-identity", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--public-corpus-plan", type=Path, required=True)
    parser.add_argument("--evaluation-plan", type=Path, required=True)
    parser.add_argument("--corrected-plan", type=Path, required=True)
    parser.add_argument("--paper-inference-addendum", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--underlying-auditor", type=Path, required=True)
    parser.add_argument("--summarizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args)
    partial = args.output.with_name(args.output.name + ".partial")
    if args.output.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    partial.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    partial.replace(args.output)
    print(json.dumps({"output": str(args.output), "passed": result["passed"],
                      "errors": result["errors"]}, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
