#!/usr/bin/env python3
"""Bind formal RULER-256K output to the corrected Qwen3-4B-Base chain."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


PROTOCOL = "in_place_ttt_qwen3_4b_base_ruler_256k_audit_v1"
MODEL_REVISION = "906bfd4b4dc7f14ee4320094d8b41684abff8539"
ARMS = ("no_ttt", "ttt")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def audit(args: argparse.Namespace) -> dict[str, Any]:
    underlying_path = args.root / "independent_audit.json"
    summary_path = args.root / "summary.json"
    metadata_path = args.root / "formal_metadata.json"
    training_path = args.training_root / "independent_audit.json"
    matrix_path = args.matrix_root / "independent_matrix_audit.json"
    resource_path = args.resource_root / "independent_resource_audit.json"
    required = [
        underlying_path, summary_path, metadata_path, training_path,
        args.corrected_training_audit, matrix_path, args.corrected_matrix_audit,
        resource_path, args.corrected_resource_audit, args.review_packet,
        args.release, args.model_identity, args.tokenizer / "tokenizer.json",
        args.tokenizer / "tokenizer_config.json", args.public_corpus_plan,
        args.evaluation_plan, args.extrapolation_plan, args.formal_plan,
        args.corrected_256k_plan, args.release_addendum, args.formal_addendum,
        args.paper_inference_addendum, args.runner, args.underlying_auditor,
        args.single_runner, args.summarizer, args.review_builder,
        args.transition_tool, args.release_tool,
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
    matrix, corrected_matrix = load(matrix_path), load(args.corrected_matrix_audit)
    resource, corrected_resource = load(resource_path), load(
        args.corrected_resource_audit
    )
    review, release, identity = (
        load(args.review_packet), load(args.release), load(args.model_identity)
    )

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
            or corrected_matrix.get("protocol")
            != "in_place_ttt_qwen3_4b_base_ruler_matrix_audit_v1"
            or corrected_matrix.get("passed") is not True
            or corrected_matrix.get("errors") != []
            or corrected_matrix.get("model_revision") != MODEL_REVISION
            or corrected_matrix.get("underlying_matrix_audit_sha256")
            != sha256(matrix_path)
            or corrected_matrix.get("corrected_training_audit_sha256")
            != sha256(args.corrected_training_audit)):
        errors.append("corrected primary RULER matrix is invalid")
    if (resource.get("protocol")
            != "in_place_ttt_2x5090_ruler_256k_resource_audit_v1"
            or resource.get("passed") is not True or resource.get("errors") != []
            or resource.get("formal_256k_released") is not True
            or corrected_resource.get("protocol")
            != "in_place_ttt_qwen3_4b_base_ruler_256k_resource_audit_v1"
            or corrected_resource.get("passed") is not True
            or corrected_resource.get("errors") != []
            or corrected_resource.get("model_revision") != MODEL_REVISION
            or corrected_resource.get("underlying_resource_audit_sha256")
            != sha256(resource_path)
            or corrected_resource.get("corrected_training_audit_sha256")
            != sha256(args.corrected_training_audit)
            or corrected_resource.get("corrected_matrix_audit_sha256")
            != sha256(args.corrected_matrix_audit)):
        errors.append("corrected 256K resource prerequisite is invalid")

    expected_review_bindings = {
        "resource_audit_sha256": sha256(resource_path),
        "corrected_resource_audit_sha256": sha256(args.corrected_resource_audit),
        "model_identity_sha256": sha256(args.model_identity),
        "release_tool_sha256": sha256(args.release_tool),
        "review_builder_sha256": sha256(args.review_builder),
        "transition_tool_sha256": sha256(args.transition_tool),
        "plan_sha256": sha256(args.release_addendum),
    }
    if (review.get("protocol")
            != "in_place_ttt_qwen3_4b_base_ruler_256k_release_review_v1"
            or review.get("ready_for_human_review") is not True
            or review.get("formal_release_created") is not False
            or review.get("target_release_path") != str(args.release.resolve())
            or review.get("bindings") != expected_review_bindings
            or set(review.get("required_human_fields", []))
            != {"accept", "reviewed_by", "decision_note"}):
        errors.append("corrected 256K review packet is invalid")
    embedded = review.get("review", {})
    if (embedded.get("resource_audit_sha256") != sha256(resource_path)
            or embedded.get("corrected_resource_audit_sha256")
            != sha256(args.corrected_resource_audit)
            or embedded.get("model_identity_sha256") != sha256(args.model_identity)
            or embedded.get("model_revision") != MODEL_REVISION):
        errors.append("corrected 256K review content is stale")
    if (release.get("protocol")
            != "in_place_ttt_2x5090_ruler_256k_formal_release_v1"
            or release.get("corrected_base_release_protocol")
            != "in_place_ttt_qwen3_4b_base_ruler_256k_formal_release_v1"
            or release.get("accepted") is not True
            or release.get("model_revision") != MODEL_REVISION
            or release.get("resource_audit_sha256") != sha256(resource_path)
            or release.get("corrected_resource_audit_sha256")
            != sha256(args.corrected_resource_audit)
            or release.get("review_packet_sha256") != sha256(args.review_packet)
            or release.get("release_transition_plan_sha256")
            != sha256(args.release_addendum)
            or not str(release.get("reviewed_by", "")).strip()
            or not str(release.get("decision_note", "")).strip()):
        errors.append("accepted corrected 256K release is invalid")

    scores = underlying.get("scores")
    if (underlying.get("protocol")
            != "in_place_ttt_2x5090_ruler_256k_audit_v1"
            or underlying.get("passed") is not True
            or underlying.get("errors") != []
            or underlying.get("context_length") != 262144
            or underlying.get("claim_boundary")
            != "public_data_yarn_extrapolation_not_paper_exact"
            or not isinstance(scores, dict)
            or set(scores) != {"no_ttt", "ttt", "ttt_minus_no_ttt"}
            or not all(isinstance(value, (int, float))
                       and not isinstance(value, bool) and math.isfinite(value)
                       for value in scores.values())):
        errors.append("underlying formal 256K audit failed")
    if (summary.get("protocol")
            != "in_place_ttt_2x5090_ruler_256k_summary_v1"
            or summary.get("completed") is not True
            or summary.get("errors") != []
            or summary.get("scores") != scores):
        errors.append("formal 256K summary differs from audited scores")

    paper_hash = sha256(args.paper_inference_addendum)
    child_bindings: dict[str, str] = {}
    for arm in ARMS:
        child_path = args.root / arm / "independent_audit.json"
        if not child_path.is_file():
            errors.append(f"{arm}: child audit is missing")
            continue
        child = load(child_path)
        if (child.get("protocol") != "in_place_ttt_2x5090_ruler_audit_v1"
                or child.get("passed") is not True or child.get("errors") != []
                or child.get("arm") != arm
                or child.get("context_length") != 262144
                or child.get("bindings", {}).get(
                    "paper_inference_addendum_sha256") != paper_hash):
            errors.append(f"{arm}: corrected child or paper overlay binding differs")
        child_bindings[arm] = sha256(child_path)

    expected_metadata = {
        "training_audit_sha256": sha256(training_path),
        "matrix_audit_sha256": sha256(matrix_path),
        "resource_audit_sha256": sha256(resource_path),
        "release_sha256": sha256(args.release),
        "public_corpus_plan_sha256": sha256(args.public_corpus_plan),
        "evaluation_plan_sha256": sha256(args.evaluation_plan),
        "extrapolation_plan_sha256": sha256(args.extrapolation_plan),
        "formal_plan_sha256": sha256(args.formal_plan),
        "runner_sha256": sha256(args.runner),
        "single_runner_sha256": sha256(args.single_runner),
        "summarizer_sha256": sha256(args.summarizer),
        "auditor_sha256": sha256(args.underlying_auditor),
    }
    if (metadata.get("protocol")
            != "in_place_ttt_2x5090_ruler_256k_formal_run_v1"
            or metadata.get("hostname") != "user-Super-Server"
            or metadata.get("training_root") != str(args.training_root.resolve())
            or metadata.get("matrix_root") != str(args.matrix_root.resolve())
            or metadata.get("resource_root") != str(args.resource_root.resolve())
            or any(metadata.get(key) != value
                   for key, value in expected_metadata.items())):
        errors.append("formal 256K metadata/dependency binding is invalid")

    return {
        "protocol": PROTOCOL,
        "passed": not errors,
        "errors": errors,
        "model_revision": MODEL_REVISION,
        "context_length": 262144,
        "extrapolation": True,
        "scores": scores,
        "underlying_formal_audit_sha256": sha256(underlying_path),
        "corrected_training_audit_sha256": sha256(args.corrected_training_audit),
        "corrected_matrix_audit_sha256": sha256(args.corrected_matrix_audit),
        "corrected_resource_audit_sha256": sha256(args.corrected_resource_audit),
        "review_packet_sha256": sha256(args.review_packet),
        "release_sha256": sha256(args.release),
        "child_audit_sha256": child_bindings,
        "bindings": expected_metadata | {
            "model_identity_sha256": sha256(args.model_identity),
            "corrected_256k_plan_sha256": sha256(args.corrected_256k_plan),
            "release_addendum_sha256": sha256(args.release_addendum),
            "formal_addendum_sha256": sha256(args.formal_addendum),
            "paper_inference_addendum_sha256": paper_hash,
        },
        "claim_boundary": "public_data_yarn_extrapolation_not_paper_exact",
        "update_admission_released": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "root", "training_root", "corrected_training_audit", "matrix_root",
        "corrected_matrix_audit", "resource_root", "corrected_resource_audit",
        "review_packet", "release", "model_identity", "tokenizer",
        "public_corpus_plan", "evaluation_plan", "extrapolation_plan",
        "formal_plan", "corrected_256k_plan", "release_addendum",
        "formal_addendum", "paper_inference_addendum", "runner",
        "underlying_auditor", "single_runner", "summarizer", "review_builder",
        "transition_tool", "release_tool", "output",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    args = argparse.Namespace(**{
        key: value.expanduser().resolve() for key, value in vars(args).items()
    })
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
