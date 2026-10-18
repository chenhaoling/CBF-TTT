#!/usr/bin/env python3
"""Bind the existing Stage-2 timing audit to the corrected Base chain."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


PROTOCOL = "in_place_ttt_qwen3_4b_base_stage2_timing_audit_v1"
MODEL_REVISION = "906bfd4b4dc7f14ee4320094d8b41684abff8539"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def audit(args: argparse.Namespace) -> dict[str, Any]:
    timing_audit_path = args.timing_root / "independent_audit.json"
    metadata_path = args.timing_root / "run_metadata.json"
    stage1_audit_path = args.stage1_root / "independent_audit.json"
    ladder_audit_path = args.ladder_root / "independent_ladder_audit.json"
    manifest_path = args.corpus_root / "manifest.json"
    data_path = args.corpus_root / "stage2_128k.jsonl"
    required = [
        timing_audit_path, metadata_path, stage1_audit_path,
        args.corrected_stage1_audit, ladder_audit_path,
        args.corrected_ladder_audit, args.model_identity, manifest_path,
        args.corpus_audit, data_path, args.timing_plan, args.resource_plan,
        args.runner, args.underlying_auditor, args.no_ttt_config,
        args.ttt_config,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return {"protocol": PROTOCOL, "passed": False,
                "errors": [f"missing required files: {missing}"]}

    errors: list[str] = []
    timing, metadata = load(timing_audit_path), load(metadata_path)
    stage1, corrected_stage1 = load(stage1_audit_path), load(
        args.corrected_stage1_audit
    )
    ladder, corrected_ladder = load(ladder_audit_path), load(
        args.corrected_ladder_audit
    )
    identity, manifest = load(args.model_identity), load(manifest_path)
    corpus_audit = load(args.corpus_audit)

    if (identity.get("passed") is not True
            or not identity.get("checks")
            or not all(identity.get("checks", {}).values())
            or identity.get("revision") != MODEL_REVISION):
        errors.append("Base identity audit is invalid")
    if (stage1.get("protocol")
            != "in_place_ttt_2x5090_100m_stage1_audit_v1"
            or stage1.get("passed") is not True or stage1.get("errors") != []
            or set(stage1.get("arms", {})) != {"no_ttt", "ttt"}):
        errors.append("underlying formal Stage-1 audit is invalid")
    if (corrected_stage1.get("protocol")
            != "in_place_ttt_qwen3_4b_base_100m_stage1_audit_v1"
            or corrected_stage1.get("passed") is not True
            or corrected_stage1.get("errors") != []
            or corrected_stage1.get("model_revision") != MODEL_REVISION
            or corrected_stage1.get("underlying_formal_audit_sha256")
            != sha256(stage1_audit_path)):
        errors.append("corrected formal Stage-1 audit is invalid or stale")
    if (ladder.get("protocol")
            != "in_place_ttt_2x5090_stage2_ladder_audit_v1"
            or ladder.get("passed") is not True or ladder.get("errors") != []
            or set(ladder.get("boundaries", {})) != {
                "65536_no_ttt", "65536_ttt",
                "131072_no_ttt", "131072_ttt",
            }):
        errors.append("underlying Stage-2 ladder audit is invalid")
    if (corrected_ladder.get("protocol")
            != "in_place_ttt_qwen3_4b_base_stage2_ladder_audit_v1"
            or corrected_ladder.get("passed") is not True
            or corrected_ladder.get("errors") != []
            or corrected_ladder.get("model_revision") != MODEL_REVISION
            or corrected_ladder.get("underlying_ladder_audit_sha256")
            != sha256(ladder_audit_path)
            or corrected_ladder.get("corrected_stage1_audit_sha256")
            != sha256(args.corrected_stage1_audit)):
        errors.append("corrected Stage-2 ladder audit is invalid or stale")

    manifest_sha = sha256(manifest_path)
    stage = next((item for item in manifest.get("stages", [])
                  if item.get("name") == "stage2_128k"), {})
    artifact = stage.get("artifacts", {}).get(data_path.name, {})
    if (manifest.get("protocol") != "in_place_ttt_public_corpus_v1"
            or manifest.get("profile") != "full"
            or corpus_audit.get("passed") is not True
            or corpus_audit.get("profile") != "full"
            or corpus_audit.get("bindings", {}).get("manifest_sha256")
            != manifest_sha
            or artifact.get("sha256") != sha256(data_path)
            or artifact.get("size_bytes") != data_path.stat().st_size):
        errors.append("corrected full Stage-2 corpus binding is invalid")

    if (timing.get("protocol")
            != "in_place_ttt_2x5090_stage2_timing_audit_v1"
            or timing.get("passed") is not True or timing.get("errors") != []
            or set(timing.get("arms", {})) != {"no_ttt", "ttt"}):
        errors.append("underlying Stage-2 timing audit failed")
    for arm, record in timing.get("arms", {}).items():
        projection = record.get("projected_formal_stage2_seconds")
        if (len(record.get("progress", [])) != 3
                or not isinstance(record.get("warm_excluded_median_seconds"),
                                  (int, float))
                or not math.isfinite(record["warm_excluded_median_seconds"])
                or record["warm_excluded_median_seconds"] <= 0
                or not isinstance(projection, (int, float))
                or not math.isfinite(projection) or projection <= 0):
            errors.append(f"{arm}: invalid corrected timing record")

    expected = {
        "timing_plan_sha256": sha256(args.timing_plan),
        "execution_addendum_sha256": sha256(args.resource_plan),
        "stage1_audit_sha256": sha256(stage1_audit_path),
        "stage2_ladder_audit_sha256": sha256(ladder_audit_path),
        "corpus_manifest_sha256": manifest_sha,
        "corpus_audit_sha256": sha256(args.corpus_audit),
        "data_sha256": sha256(data_path),
        "no_ttt_config_sha256": sha256(args.no_ttt_config),
        "ttt_config_sha256": sha256(args.ttt_config),
        "runner_sha256": sha256(args.runner),
        "auditor_sha256": sha256(args.underlying_auditor),
    }
    if (metadata.get("protocol") != "in_place_ttt_2x5090_stage2_timing_v1"
            or metadata.get("hostname") != "user-Super-Server"
            or metadata.get("stage1_root") != str(args.stage1_root.resolve())
            or metadata.get("stage2_ladder_root")
            != str(args.ladder_root.resolve())
            or metadata.get("steps") != 3
            or metadata.get("sequence_length") != 131_072
            or metadata.get("global_batch_size") != 64
            or metadata.get("tokens_per_arm") != 25_165_824
            or any(metadata.get(key) != value for key, value in expected.items())):
        errors.append("corrected Stage-2 timing metadata binding mismatch")

    paired = timing.get("projected_sequential_stage2_seconds")
    if not isinstance(paired, (int, float)) or not math.isfinite(paired) \
            or paired <= 0:
        errors.append("paired formal Stage-2 projection is invalid")
    return {
        "protocol": PROTOCOL,
        "passed": not errors,
        "errors": errors,
        "model_revision": MODEL_REVISION,
        "underlying_timing_audit_sha256": sha256(timing_audit_path),
        "corrected_stage1_audit_sha256": sha256(args.corrected_stage1_audit),
        "underlying_ladder_audit_sha256": sha256(ladder_audit_path),
        "corrected_ladder_audit_sha256": sha256(args.corrected_ladder_audit),
        "projected_sequential_stage2_seconds": paired,
        "arms": timing.get("arms", {}),
        "bindings": expected,
        "formal_release_created": False,
        "claim_boundary": (
            "Corrected Base Stage-2 timing/resource evidence only; formal "
            "Stage 2 requires a separate accepted release."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing-root", type=Path, required=True)
    parser.add_argument("--stage1-root", type=Path, required=True)
    parser.add_argument("--corrected-stage1-audit", type=Path, required=True)
    parser.add_argument("--ladder-root", type=Path, required=True)
    parser.add_argument("--corrected-ladder-audit", type=Path, required=True)
    parser.add_argument("--model-identity", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--corpus-audit", type=Path, required=True)
    parser.add_argument("--timing-plan", type=Path, required=True)
    parser.add_argument("--resource-plan", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--underlying-auditor", type=Path, required=True)
    parser.add_argument("--no-ttt-config", type=Path, required=True)
    parser.add_argument("--ttt-config", type=Path, required=True)
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
