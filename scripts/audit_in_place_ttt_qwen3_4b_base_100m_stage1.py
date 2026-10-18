#!/usr/bin/env python3
"""Bind the formal Stage-1 paired audit to the corrected Base chain."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PROTOCOL = "in_place_ttt_qwen3_4b_base_100m_stage1_audit_v1"
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
    manifest_path = args.corpus_root / "manifest.json"
    data_path = args.corpus_root / "stage1_32k.jsonl"
    underlying_path = args.root / "independent_audit.json"
    metadata_path = args.root / "run_metadata.json"
    timing_path = args.timing_root / "independent_audit.json"
    required = [
        underlying_path, metadata_path, timing_path,
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
    timing, corrected = load(timing_path), load(args.corrected_timing_audit)
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
    manifest_sha = sha256(manifest_path)
    assets = manifest.get("bindings", {}).get("tokenizer_assets", {})
    if (manifest.get("profile") != "full"
            or manifest.get("totals", {}).get("records") != 724_800
            or manifest.get("totals", {}).get("training_tokens")
            != 34_999_369_728
            or corpus_audit.get("passed") is not True
            or not corpus_audit.get("checks")
            or not all(corpus_audit.get("checks", {}).values())
            or corpus_audit.get("bindings", {}).get("manifest_sha256")
            != manifest_sha
            or any(assets.get(name, {}).get("sha256")
                   != identity_files.get(name, {}).get("sha256")
                   for name in ("config.json", "tokenizer.json",
                                "tokenizer_config.json"))):
        errors.append("corrected full corpus binding is invalid")
    if (timing.get("protocol")
            != "in_place_ttt_2x5090_stage1_timing_audit_v1"
            or timing.get("passed") is not True or timing.get("errors") != []):
        errors.append("underlying timing audit is invalid")
    if (corrected.get("protocol")
            != "in_place_ttt_qwen3_4b_base_stage1_timing_audit_v1"
            or corrected.get("passed") is not True
            or corrected.get("errors") != []
            or corrected.get("model_revision") != MODEL_REVISION
            or corrected.get("underlying_timing_audit_sha256")
            != sha256(timing_path)):
        errors.append("corrected timing audit is invalid or stale")
    if (release.get("protocol")
            != "in_place_ttt_2x5090_100m_stage1_release_v1"
            or release.get("accepted") is not True
            or release.get("stage") != "stage1"
            or release.get("timing_audit_sha256") != sha256(timing_path)
            or release.get("corrected_timing_audit_sha256")
            != sha256(args.corrected_timing_audit)
            or release.get("model_revision") != MODEL_REVISION
            or not str(release.get("reviewed_by", "")).strip()
            or not str(release.get("decision_note", "")).strip()):
        errors.append("corrected explicit Stage-1 release is invalid")
    if (underlying.get("protocol")
            != "in_place_ttt_2x5090_100m_stage1_audit_v1"
            or underlying.get("passed") is not True
            or underlying.get("errors") != []
            or set(underlying.get("arms", {})) != {"no_ttt", "ttt"}):
        errors.append("underlying formal Stage-1 audit failed")
    expected = {
        "formal_plan_sha256": sha256(args.formal_plan),
        "execution_addendum_sha256": sha256(args.execution_addendum),
        "timing_audit_sha256": sha256(timing_path),
        "release_sha256": sha256(args.release),
        "corpus_manifest_sha256": manifest_sha,
        "corpus_audit_sha256": sha256(args.corpus_audit),
        "data_sha256": sha256(data_path),
        "model_config_sha256": identity_files.get("config.json", {}).get("sha256"),
        "no_ttt_config_sha256": sha256(args.no_ttt_config),
        "ttt_config_sha256": sha256(args.ttt_config),
        "runner_sha256": sha256(args.runner),
        "auditor_sha256": sha256(args.underlying_auditor),
    }
    if (metadata.get("protocol") != "in_place_ttt_2x5090_100m_stage1_v1"
            or metadata.get("hostname") != "user-Super-Server"
            or metadata.get("steps_per_arm") != 27
            or metadata.get("sequence_length") != 32768
            or metadata.get("global_batch_size") != 64
            or metadata.get("tokens_per_arm") != 56_623_104
            or any(metadata.get(key) != value for key, value in expected.items())):
        errors.append("corrected formal Stage-1 metadata binding mismatch")
    return {
        "protocol": PROTOCOL,
        "passed": not errors,
        "errors": errors,
        "model_revision": MODEL_REVISION,
        "underlying_formal_audit_sha256": sha256(underlying_path),
        "underlying_timing_audit_sha256": sha256(timing_path),
        "corrected_timing_audit_sha256": sha256(args.corrected_timing_audit),
        "release_sha256": sha256(args.release),
        "arms": underlying.get("arms", {}),
        "bindings": expected,
        "claim_boundary": (
            "Corrected Base public-data 100M-tier Stage-1 only; Stage 2, "
            "RULER and update-admission remain unreleased."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-identity", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--corpus-audit", type=Path, required=True)
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
    if args.output.exists() or args.output.with_name(
            args.output.name + ".partial").exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    partial = args.output.with_name(args.output.name + ".partial")
    partial.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    partial.replace(args.output)
    print(json.dumps({"output": str(args.output), "passed": result["passed"],
                      "errors": result["errors"]}, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
