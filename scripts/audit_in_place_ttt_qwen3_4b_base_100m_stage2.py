#!/usr/bin/env python3
"""Bind formal Stage-2 paired evidence to the corrected Base chain."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PROTOCOL = "in_place_ttt_qwen3_4b_base_100m_stage2_audit_v1"
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
    underlying_path = args.root / "independent_audit.json"
    metadata_path = args.root / "run_metadata.json"
    stage1_path = args.stage1_root / "independent_audit.json"
    ladder_path = args.ladder_root / "independent_ladder_audit.json"
    timing_path = args.timing_root / "independent_audit.json"
    manifest_path = args.corpus_root / "manifest.json"
    data_path = args.corpus_root / "stage2_128k.jsonl"
    required = [
        underlying_path, metadata_path, stage1_path,
        args.corrected_stage1_audit, ladder_path,
        args.corrected_ladder_audit, timing_path,
        args.corrected_timing_audit, args.model_identity, manifest_path,
        args.corpus_audit, data_path, args.release, args.formal_plan,
        args.execution_addendum, args.runner, args.underlying_auditor,
        args.no_ttt_config, args.ttt_config,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return {"protocol": PROTOCOL, "passed": False,
                "errors": [f"missing required files: {missing}"]}

    errors: list[str] = []
    underlying, metadata = load(underlying_path), load(metadata_path)
    stage1, corrected_stage1 = load(stage1_path), load(
        args.corrected_stage1_audit
    )
    ladder, corrected_ladder = load(ladder_path), load(
        args.corrected_ladder_audit
    )
    timing, corrected_timing = load(timing_path), load(
        args.corrected_timing_audit
    )
    identity, manifest = load(args.model_identity), load(manifest_path)
    corpus_audit, release = load(args.corpus_audit), load(args.release)
    identity_files = identity.get("files", {})

    if (identity.get("passed") is not True
            or not identity.get("checks")
            or not all(identity.get("checks", {}).values())
            or identity.get("revision") != MODEL_REVISION):
        errors.append("Base identity audit is invalid")
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        path, saved = args.model / name, identity_files.get(name, {})
        if (not path.is_file() or path.stat().st_size != saved.get("size")
                or sha256(path) != saved.get("sha256")):
            errors.append(f"Base model asset drift: {name}")

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
            != sha256(stage1_path)):
        errors.append("corrected formal Stage-1 audit is invalid or stale")
    if (ladder.get("protocol")
            != "in_place_ttt_2x5090_stage2_ladder_audit_v1"
            or ladder.get("passed") is not True or ladder.get("errors") != []):
        errors.append("underlying Stage-2 ladder audit is invalid")
    if (corrected_ladder.get("protocol")
            != "in_place_ttt_qwen3_4b_base_stage2_ladder_audit_v1"
            or corrected_ladder.get("passed") is not True
            or corrected_ladder.get("errors") != []
            or corrected_ladder.get("underlying_ladder_audit_sha256")
            != sha256(ladder_path)
            or corrected_ladder.get("corrected_stage1_audit_sha256")
            != sha256(args.corrected_stage1_audit)):
        errors.append("corrected Stage-2 ladder audit is invalid or stale")
    if (timing.get("protocol")
            != "in_place_ttt_2x5090_stage2_timing_audit_v1"
            or timing.get("passed") is not True or timing.get("errors") != []):
        errors.append("underlying Stage-2 timing audit is invalid")
    if (corrected_timing.get("protocol")
            != "in_place_ttt_qwen3_4b_base_stage2_timing_audit_v1"
            or corrected_timing.get("passed") is not True
            or corrected_timing.get("errors") != []
            or corrected_timing.get("model_revision") != MODEL_REVISION
            or corrected_timing.get("underlying_timing_audit_sha256")
            != sha256(timing_path)
            or corrected_timing.get("corrected_ladder_audit_sha256")
            != sha256(args.corrected_ladder_audit)):
        errors.append("corrected Stage-2 timing audit is invalid or stale")
    if (release.get("protocol")
            != "in_place_ttt_2x5090_100m_stage2_release_v1"
            or release.get("accepted") is not True
            or release.get("stage") != "stage2"
            or release.get("timing_audit_sha256") != sha256(timing_path)
            or release.get("corrected_timing_audit_sha256")
            != sha256(args.corrected_timing_audit)
            or release.get("model_revision") != MODEL_REVISION
            or not str(release.get("reviewed_by", "")).strip()
            or not str(release.get("decision_note", "")).strip()):
        errors.append("corrected explicit Stage-2 release is invalid")

    manifest_sha = sha256(manifest_path)
    assets = manifest.get("bindings", {}).get("tokenizer_assets", {})
    stage = next((item for item in manifest.get("stages", [])
                  if item.get("name") == "stage2_128k"), {})
    artifact = stage.get("artifacts", {}).get(data_path.name, {})
    if (manifest.get("profile") != "full"
            or corpus_audit.get("passed") is not True
            or not corpus_audit.get("checks")
            or not all(corpus_audit.get("checks", {}).values())
            or corpus_audit.get("bindings", {}).get("manifest_sha256")
            != manifest_sha
            or artifact.get("sha256") != sha256(data_path)
            or artifact.get("size_bytes") != data_path.stat().st_size
            or any(assets.get(name, {}).get("sha256")
                   != identity_files.get(name, {}).get("sha256")
                   for name in ("config.json", "tokenizer.json",
                                "tokenizer_config.json"))):
        errors.append("corrected full Stage-2 corpus binding is invalid")

    if (underlying.get("protocol")
            != "in_place_ttt_2x5090_100m_stage2_audit_v1"
            or underlying.get("passed") is not True
            or underlying.get("errors") != []
            or set(underlying.get("arms", {})) != {"no_ttt", "ttt"}):
        errors.append("underlying formal Stage-2 paired audit failed")
    for arm, record in underlying.get("arms", {}).items():
        if (len(record.get("progress", [])) != 5
                or not record.get("dcp_manifest")
                or not record.get("hf_manifest")):
            errors.append(f"{arm}: incomplete formal Stage-2 evidence")

    expected = {
        "formal_plan_sha256": sha256(args.formal_plan),
        "execution_addendum_sha256": sha256(args.execution_addendum),
        "timing_audit_sha256": sha256(timing_path),
        "release_sha256": sha256(args.release),
        "stage1_audit_sha256": sha256(stage1_path),
        "stage2_ladder_audit_sha256": sha256(ladder_path),
        "corpus_manifest_sha256": manifest_sha,
        "corpus_audit_sha256": sha256(args.corpus_audit),
        "data_sha256": sha256(data_path),
        "no_ttt_config_sha256": sha256(args.no_ttt_config),
        "ttt_config_sha256": sha256(args.ttt_config),
        "runner_sha256": sha256(args.runner),
        "auditor_sha256": sha256(args.underlying_auditor),
    }
    if (metadata.get("protocol") != "in_place_ttt_2x5090_100m_stage2_v1"
            or metadata.get("hostname") != "user-Super-Server"
            or metadata.get("steps_per_arm") != 5
            or metadata.get("sequence_length") != 131_072
            or metadata.get("global_batch_size") != 64
            or metadata.get("tokens_per_arm") != 41_943_040
            or metadata.get("stage1_root") != str(args.stage1_root.resolve())
            or metadata.get("stage2_ladder_root")
            != str(args.ladder_root.resolve())
            or any(metadata.get(key) != value for key, value in expected.items())):
        errors.append("corrected formal Stage-2 metadata binding mismatch")
    return {
        "protocol": PROTOCOL,
        "passed": not errors,
        "errors": errors,
        "model_revision": MODEL_REVISION,
        "underlying_formal_audit_sha256": sha256(underlying_path),
        "corrected_stage1_audit_sha256": sha256(args.corrected_stage1_audit),
        "corrected_ladder_audit_sha256": sha256(args.corrected_ladder_audit),
        "underlying_timing_audit_sha256": sha256(timing_path),
        "corrected_timing_audit_sha256": sha256(args.corrected_timing_audit),
        "release_sha256": sha256(args.release),
        "arms": underlying.get("arms", {}),
        "bindings": expected,
        "claim_boundary": (
            "Corrected Base public-data 100M-tier Stage-2 only; paper-facing "
            "RULER remains separately gated."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-identity", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--corpus-audit", type=Path, required=True)
    parser.add_argument("--stage1-root", type=Path, required=True)
    parser.add_argument("--corrected-stage1-audit", type=Path, required=True)
    parser.add_argument("--ladder-root", type=Path, required=True)
    parser.add_argument("--corrected-ladder-audit", type=Path, required=True)
    parser.add_argument("--timing-root", type=Path, required=True)
    parser.add_argument("--corrected-timing-audit", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--formal-plan", type=Path, required=True)
    parser.add_argument("--execution-addendum", type=Path, required=True)
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
