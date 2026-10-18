#!/usr/bin/env python3
"""Bind the existing Stage-1 timing audit to the corrected Base chain."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


PROTOCOL = "in_place_ttt_qwen3_4b_base_stage1_timing_audit_v1"
MODEL_REVISION = "906bfd4b4dc7f14ee4320094d8b41684abff8539"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def audit(args: argparse.Namespace) -> dict[str, Any]:
    required = [
        args.timing_root / "independent_audit.json",
        args.timing_root / "run_metadata.json",
        args.model_identity,
        args.corpus_root / "manifest.json", args.corpus_audit,
        args.ladder_audit, args.timing_plan, args.stage1_plan, args.runner,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return {"protocol": PROTOCOL, "passed": False,
                "errors": [f"missing required files: {missing}"]}
    errors: list[str] = []
    timing_audit = load(args.timing_root / "independent_audit.json")
    metadata = load(args.timing_root / "run_metadata.json")
    identity = load(args.model_identity)
    manifest_path = args.corpus_root / "manifest.json"
    manifest, corpus_audit = load(manifest_path), load(args.corpus_audit)
    ladder = load(args.ladder_audit)
    identity_files = identity.get("files", {})
    if (identity.get("passed") is not True
            or not all(identity.get("checks", {}).values())
            or identity.get("revision") != MODEL_REVISION):
        errors.append("Base identity audit is invalid")
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        path, saved = args.model / name, identity_files.get(name, {})
        if (not path.is_file() or path.stat().st_size != saved.get("size")
                or sha256(path) != saved.get("sha256")):
            errors.append(f"Base model asset drift: {name}")
    manifest_sha = sha256(manifest_path)
    assets = manifest.get("bindings", {}).get("tokenizer_assets", {})
    if (manifest.get("profile") != "full"
            or manifest.get("totals", {}).get("records") != 724_800
            or manifest.get("totals", {}).get("training_tokens") != 34_999_369_728
            or corpus_audit.get("passed") is not True
            or not all(corpus_audit.get("checks", {}).values())
            or corpus_audit.get("bindings", {}).get("manifest_sha256") != manifest_sha
            or any(assets.get(name, {}).get("sha256")
                   != identity_files.get(name, {}).get("sha256")
                   for name in ("config.json", "tokenizer.json",
                                "tokenizer_config.json"))):
        errors.append("corrected full corpus binding is invalid")
    if (ladder.get("protocol")
            != "in_place_ttt_qwen3_4b_base_stage1_ladder_audit_v1"
            or ladder.get("passed") is not True
            or set(ladder.get("boundaries", {}))
            != {"16384_no_ttt", "16384_ttt", "32768_no_ttt", "32768_ttt"}):
        errors.append("corrected Stage-1 ladder did not release timing")
    if (timing_audit.get("protocol")
            != "in_place_ttt_2x5090_stage1_timing_audit_v1"
            or timing_audit.get("passed") is not True
            or set(timing_audit.get("arms", {})) != {"no_ttt", "ttt"}):
        errors.append("underlying timing audit failed")
    for arm, record in timing_audit.get("arms", {}).items():
        if (len(record.get("progress", [])) != 3
                or not isinstance(record.get("warm_excluded_median_seconds"),
                                  (int, float))
                or not math.isfinite(record["warm_excluded_median_seconds"])
                or record["warm_excluded_median_seconds"] <= 0):
            errors.append(f"{arm}: invalid corrected timing record")
    expected = {
        "timing_plan_sha256": sha256(args.timing_plan),
        "execution_addendum_sha256": sha256(args.stage1_plan),
        "ladder_audit_sha256": sha256(args.ladder_audit),
        "corpus_manifest_sha256": manifest_sha,
        "corpus_audit_sha256": sha256(args.corpus_audit),
        "model_config_sha256": identity_files["config.json"]["sha256"],
        "runner_sha256": sha256(args.runner),
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        errors.append("corrected timing dependency binding mismatch")
    projection = timing_audit.get("projected_sequential_stage1_seconds")
    if not isinstance(projection, (int, float)) or not math.isfinite(projection) \
            or projection <= 0:
        errors.append("paired formal Stage-1 projection is invalid")
    return {
        "protocol": PROTOCOL,
        "passed": not errors,
        "errors": errors,
        "model_revision": MODEL_REVISION,
        "underlying_timing_audit_sha256":
            sha256(args.timing_root / "independent_audit.json"),
        "projected_sequential_stage1_seconds": projection,
        "arms": timing_audit.get("arms", {}),
        "bindings": expected,
        "formal_release_created": False,
        "claim_boundary": (
            "Corrected Base Stage-1 timing/resource evidence only; formal "
            "training requires a separate accepted release."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-identity", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--corpus-audit", type=Path, required=True)
    parser.add_argument("--ladder-audit", type=Path, required=True)
    parser.add_argument("--timing-plan", type=Path, required=True)
    parser.add_argument("--stage1-plan", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "passed": result["passed"],
                      "errors": result["errors"]}, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
