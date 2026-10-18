#!/usr/bin/env python3
"""Bind the 256K resource result to the corrected Qwen3-4B-Base chain."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PROTOCOL = "in_place_ttt_qwen3_4b_base_ruler_256k_resource_audit_v1"
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
    resource_path = args.root / "independent_resource_audit.json"
    metadata_path = args.root / "resource_metadata.json"
    training_path = args.training_root / "independent_audit.json"
    matrix_path = args.matrix_root / "independent_matrix_audit.json"
    required = [
        resource_path, metadata_path, training_path,
        args.corrected_training_audit, matrix_path, args.corrected_matrix_audit,
        args.model_identity, args.tokenizer / "tokenizer.json",
        args.tokenizer / "tokenizer_config.json", args.public_corpus_plan,
        args.evaluation_plan, args.extrapolation_plan, args.corrected_primary_plan,
        args.corrected_256k_plan, args.paper_inference_addendum, args.runner,
        args.underlying_auditor, args.single_runner,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return {"protocol": PROTOCOL, "passed": False,
                "errors": [f"missing required files: {missing}"]}

    errors: list[str] = []
    resource, metadata = load(resource_path), load(metadata_path)
    training, corrected_training = load(training_path), load(
        args.corrected_training_audit
    )
    matrix, corrected_matrix = load(matrix_path), load(args.corrected_matrix_audit)
    identity = load(args.model_identity)
    identity_files = identity.get("files", {})
    if (identity.get("passed") is not True or not identity.get("checks")
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
    if (matrix.get("protocol") != "in_place_ttt_2x5090_ruler_matrix_audit_v1"
            or matrix.get("passed") is not True or matrix.get("errors") != []
            or set(matrix.get("recomputed_context_scores", {})) != CONTEXTS
            or corrected_matrix.get("protocol")
            != "in_place_ttt_qwen3_4b_base_ruler_matrix_audit_v1"
            or corrected_matrix.get("passed") is not True
            or corrected_matrix.get("errors") != []
            or corrected_matrix.get("model_revision") != MODEL_REVISION
            or corrected_matrix.get("underlying_matrix_audit_sha256")
            != sha256(matrix_path)
            or corrected_matrix.get("corrected_training_audit_sha256")
            != sha256(args.corrected_training_audit)):
        errors.append("corrected 4K-128K RULER matrix prerequisite is invalid")
    if (resource.get("protocol")
            != "in_place_ttt_2x5090_ruler_256k_resource_audit_v1"
            or resource.get("passed") is not True or resource.get("errors") != []
            or resource.get("context_length") != 262144
            or resource.get("extrapolation") is not True
            or resource.get("scores_discarded") is not True
            or resource.get("formal_256k_released") is not True
            or set(resource.get("boundaries", {})) != {"no_ttt", "ttt"}):
        errors.append("underlying 256K resource audit failed")

    expected = {
        "training_audit_sha256": sha256(training_path),
        "matrix_audit_sha256": sha256(matrix_path),
        "public_corpus_plan_sha256": sha256(args.public_corpus_plan),
        "evaluation_plan_sha256": sha256(args.evaluation_plan),
        "extrapolation_plan_sha256": sha256(args.extrapolation_plan),
        "runner_sha256": sha256(args.runner),
        "single_runner_sha256": sha256(args.single_runner),
        "auditor_sha256": sha256(args.underlying_auditor),
    }
    if (metadata.get("protocol")
            != "in_place_ttt_2x5090_ruler_256k_resource_run_v1"
            or metadata.get("hostname") != "user-Super-Server"
            or metadata.get("training_root") != str(args.training_root.resolve())
            or metadata.get("matrix_root") != str(args.matrix_root.resolve())
            or metadata.get("context_length") != 262144
            or any(metadata.get(key) != value for key, value in expected.items())):
        errors.append("256K resource metadata binding mismatch")
    bindings = resource.get("bindings", {})
    if any(bindings.get(key) != value for key, value in expected.items()):
        errors.append("underlying 256K resource binding is stale")
    paper_hash = sha256(args.paper_inference_addendum)
    for arm, boundary in resource.get("boundaries", {}).items():
        if boundary.get("bindings", {}).get(
                "paper_inference_addendum_sha256") != paper_hash:
            errors.append(f"{arm}: paper inference clipping binding differs")

    return {
        "protocol": PROTOCOL,
        "passed": not errors,
        "errors": errors,
        "model_revision": MODEL_REVISION,
        "context_length": 262144,
        "extrapolation": True,
        "scores_discarded": True,
        "formal_256k_released": not errors,
        "underlying_resource_audit_sha256": sha256(resource_path),
        "corrected_training_audit_sha256": sha256(args.corrected_training_audit),
        "corrected_matrix_audit_sha256": sha256(args.corrected_matrix_audit),
        "boundaries": resource.get("boundaries", {}),
        "bindings": expected | {
            "model_identity_sha256": sha256(args.model_identity),
            "corrected_primary_plan_sha256": sha256(args.corrected_primary_plan),
            "corrected_256k_plan_sha256": sha256(args.corrected_256k_plan),
            "paper_inference_addendum_sha256": paper_hash,
        },
        "claim_boundary": "public_data_yarn_extrapolation_resource_only",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--corrected-training-audit", type=Path, required=True)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--corrected-matrix-audit", type=Path, required=True)
    parser.add_argument("--model-identity", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--public-corpus-plan", type=Path, required=True)
    parser.add_argument("--evaluation-plan", type=Path, required=True)
    parser.add_argument("--extrapolation-plan", type=Path, required=True)
    parser.add_argument("--corrected-primary-plan", type=Path, required=True)
    parser.add_argument("--corrected-256k-plan", type=Path, required=True)
    parser.add_argument("--paper-inference-addendum", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--underlying-auditor", type=Path, required=True)
    parser.add_argument("--single-runner", type=Path, required=True)
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
