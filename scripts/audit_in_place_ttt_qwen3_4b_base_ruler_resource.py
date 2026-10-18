#!/usr/bin/env python3
"""Bind the RULER resource ladder to corrected Base Stage-2 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PROTOCOL = "in_place_ttt_qwen3_4b_base_ruler_resource_audit_v1"
MODEL_REVISION = "906bfd4b4dc7f14ee4320094d8b41684abff8539"
EXPECTED_BOUNDARIES = {
    f"{suffix}_{arm}" for suffix in ("4k", "32k", "64k", "128k")
    for arm in ("no_ttt", "ttt")
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def audit(args: argparse.Namespace) -> dict[str, Any]:
    underlying_path = args.root / "independent_resource_ladder_audit.json"
    metadata_path = args.root / "ladder_metadata.json"
    training_path = args.training_root / "independent_audit.json"
    required = [
        underlying_path, metadata_path, training_path,
        args.corrected_training_audit, args.model_identity,
        args.tokenizer / "tokenizer.json",
        args.tokenizer / "tokenizer_config.json", args.public_corpus_plan,
        args.evaluation_plan, args.corrected_plan,
        args.paper_inference_addendum, args.runner,
        args.underlying_auditor, args.single_runner,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return {"protocol": PROTOCOL, "passed": False,
                "errors": [f"missing required files: {missing}"]}

    errors: list[str] = []
    underlying, metadata = load(underlying_path), load(metadata_path)
    training, corrected = load(training_path), load(
        args.corrected_training_audit
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
            or set(training.get("arms", {})) != {"no_ttt", "ttt"}):
        errors.append("underlying formal Stage-2 audit is invalid")
    if (corrected.get("protocol")
            != "in_place_ttt_qwen3_4b_base_100m_stage2_audit_v1"
            or corrected.get("passed") is not True
            or corrected.get("errors") != []
            or corrected.get("model_revision") != MODEL_REVISION
            or corrected.get("underlying_formal_audit_sha256")
            != sha256(training_path)):
        errors.append("corrected formal Stage-2 audit is invalid or stale")
    if (underlying.get("protocol")
            != "in_place_ttt_2x5090_ruler_resource_ladder_audit_v1"
            or underlying.get("passed") is not True
            or underlying.get("errors") != []
            or underlying.get("scores_discarded") is not True
            or set(underlying.get("boundaries", {})) != EXPECTED_BOUNDARIES):
        errors.append("underlying RULER resource ladder audit failed")

    expected = {
        "training_audit_sha256": sha256(training_path),
        "public_corpus_plan_sha256": sha256(args.public_corpus_plan),
        "evaluation_plan_sha256": sha256(args.evaluation_plan),
        "paper_inference_addendum_sha256": sha256(
            args.paper_inference_addendum
        ),
        "runner_sha256": sha256(args.runner),
        "single_runner_sha256": sha256(args.single_runner),
        "auditor_sha256": sha256(args.underlying_auditor),
    }
    if (metadata.get("protocol")
            != "in_place_ttt_2x5090_ruler_resource_ladder_run_v1"
            or metadata.get("hostname") != "user-Super-Server"
            or metadata.get("training_root")
            != str(args.training_root.resolve())
            or metadata.get("boundaries") != [4096, 32768, 65536, 131072]
            or any(metadata.get(key) != value for key, value in expected.items())):
        errors.append("corrected RULER resource metadata binding mismatch")
    bindings = underlying.get("bindings", {})
    if any(bindings.get(key) != value for key, value in expected.items()):
        errors.append("underlying RULER resource audit binding is stale")
    return {
        "protocol": PROTOCOL,
        "passed": not errors,
        "errors": errors,
        "model_revision": MODEL_REVISION,
        "underlying_resource_audit_sha256": sha256(underlying_path),
        "underlying_training_audit_sha256": sha256(training_path),
        "corrected_training_audit_sha256": sha256(
            args.corrected_training_audit
        ),
        "boundaries": underlying.get("boundaries", {}),
        "scores_discarded": True,
        "bindings": expected | {
            "corrected_plan_sha256": sha256(args.corrected_plan),
            "model_identity_sha256": sha256(args.model_identity),
        },
        "claim_boundary": (
            "Corrected Base RULER resource evidence only; all scores are "
            "discarded and cannot support an efficacy claim."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--corrected-training-audit", type=Path, required=True)
    parser.add_argument("--model-identity", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--public-corpus-plan", type=Path, required=True)
    parser.add_argument("--evaluation-plan", type=Path, required=True)
    parser.add_argument("--corrected-plan", type=Path, required=True)
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
