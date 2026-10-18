#!/usr/bin/env python3
"""Independent checks for a paper-clipped inference overlay in a RULER run."""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
from typing import Any, Optional


EXPECTED_SOURCE = {
    "inference_model/__init__.py": (648, "54ae124b5c993dff594b041dc4375407fbf1c016a689f9dac805e0043e86e52a"),
    "inference_model/hf_llama3/__init__.py": (965, "f084e24edbc2bb8cebe74f3ded368d40e1c6a848274f06a8077d09135567a1c9"),
    "inference_model/hf_llama3/configuration_llama.py": (12819, "13e03dfcc525f809b215a0effae2837c82da612203f40d645b449e740650f8e8"),
    "inference_model/hf_llama3/modeling_llama.py": (28170, "78e95c6cc3a64d4fc629ca6a9abb135883d1c0a6ca5184a2a0b8283f06802d85"),
    "inference_model/hf_qwen3/__init__.py": (965, "a38d76873e560e9c034258408e1c67fdfc08f2be846da1937cea1434f6e10e80"),
    "inference_model/hf_qwen3/configuration_qwen3.py": (12631, "39ef69d7c79114ae9bd2710320c4260796fc5dbed9f725e18b5a0f5771114ec2"),
    "inference_model/hf_qwen3/modeling_qwen3.py": (29991, "98ec94bcf1780635d7817b993a0bf4fa1f1a7e3a35415dce8a46c345a52a2858"),
}
EXPECTED_OVERLAY = {
    **EXPECTED_SOURCE,
    "inference_model/hf_qwen3/configuration_qwen3.py": (13000, "c23540e8c00d659cef9cb810df7f75efa3389662736bf5570bc1d8f61bab86a6"),
    "inference_model/hf_qwen3/modeling_qwen3.py": (30724, "a212c35ca9a5471f8ff9a1ef70c9cb0ec452c87eaac50dd4da51c8ce8d045103"),
}
MODIFIED_FILES = [
    "inference_model/hf_qwen3/configuration_qwen3.py",
    "inference_model/hf_qwen3/modeling_qwen3.py",
]


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative(root: pathlib.Path, relative: Any) -> Optional[pathlib.Path]:
    if not isinstance(relative, str):
        return None
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return None
    return path


def _manifest_records(expected: dict[str, tuple[int, str]]) -> dict[str, dict]:
    return {
        relative: {"size_bytes": size, "sha256": digest}
        for relative, (size, digest) in expected.items()
    }


def audit_paper_inference_overlay(
    root: pathlib.Path,
    metadata: dict,
    addendum: pathlib.Path,
    tool_root: pathlib.Path,
) -> list[str]:
    errors: list[str] = []
    if not addendum.is_file() or metadata.get("paper_inference_addendum_sha256") != sha256(addendum):
        errors.append("paper inference addendum binding differs")
    if (metadata.get("paper_inference_semantics")
            != "per_update_delta_frobenius_clip"
            or metadata.get("ttt_update_clip_norm") != 1e-5):
        errors.append("paper inference clip semantics differ")

    expected_manifest_rel = "paper_inference_overlay/paper_inference_overlay_manifest.json"
    expected_validation_rel = "paper_inference_overlay_validation.json"
    if metadata.get("paper_inference_overlay_manifest_relative_path") != expected_manifest_rel:
        errors.append("paper inference overlay manifest path differs")
    if metadata.get("paper_inference_overlay_validation_relative_path") != expected_validation_rel:
        errors.append("paper inference overlay validation path differs")
    manifest_path = _safe_relative(root, expected_manifest_rel)
    validation_path = _safe_relative(root, expected_validation_rel)
    if manifest_path is None or not manifest_path.is_file():
        errors.append("paper inference overlay manifest is missing")
        return errors
    if validation_path is None or not validation_path.is_file():
        errors.append("paper inference overlay validation is missing")
        return errors
    if metadata.get("paper_inference_overlay_manifest_sha256") != sha256(manifest_path):
        errors.append("paper inference overlay manifest hash differs")
    if metadata.get("paper_inference_overlay_validation_sha256") != sha256(validation_path):
        errors.append("paper inference overlay validation hash differs")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        errors.append(f"paper inference JSON is invalid: {exc}")
        return errors
    if (manifest.get("protocol") != "in_place_ttt_paper_inference_overlay_v1"
            or manifest.get("upstream_commit")
            != "be2324829b0e91c8fd10a74d4b43714fde6676e1"
            or manifest.get("clip_scope")
            != "each_native_qwen3_fast_weight_delta"
            or manifest.get("clip_norm_type") != "frobenius"
            or manifest.get("clip_norm") != 1e-5
            or manifest.get("default_without_override") is not None
            or manifest.get("modified_files") != MODIFIED_FILES):
        errors.append("paper inference overlay manifest semantics differ")
    if manifest.get("source_files") != _manifest_records(EXPECTED_SOURCE):
        errors.append("paper inference upstream source manifest differs")
    if manifest.get("overlay_files") != _manifest_records(EXPECTED_OVERLAY):
        errors.append("paper inference generated file manifest differs")

    overlay_root = manifest_path.parent
    actual_files = {
        str(path.relative_to(overlay_root)): (path.stat().st_size, sha256(path))
        for path in overlay_root.rglob("*.py") if path.is_file()
    }
    if actual_files != EXPECTED_OVERLAY:
        errors.append("paper inference generated files differ on disk")
    builder = tool_root / "scripts/build_in_place_ttt_paper_inference_overlay.py"
    if not builder.is_file() or manifest.get("builder_sha256") != sha256(builder):
        errors.append("paper inference builder binding differs")

    clipped = validation.get("clipped_delta_frobenius_norm")
    unclipped = validation.get("unclipped_delta_frobenius_norm")
    if (validation.get("protocol")
            != "in_place_ttt_paper_inference_overlay_validation_v1"
            or validation.get("passed") is not True
            or validation.get("device") != "cpu"
            or validation.get("cuda_initialized") is not False
            or validation.get("clip_norm") != 1e-5
            or validation.get("zero_delta_unchanged") is not True
            or validation.get("config_default_disabled") is not True
            or validation.get("config_round_trip") is not True
            or validation.get("from_pretrained_override") is not True
            or validation.get("invalid_values_rejected")
            != ["0.0", "-1.0", "inf", "nan"]
            or not isinstance(clipped, (int, float))
            or not math.isclose(clipped, 1e-5, rel_tol=2e-5, abs_tol=1e-10)
            or not isinstance(unclipped, (int, float))
            or not unclipped > 1e-5):
        errors.append("paper inference CPU numerical validation differs")
    if validation.get("overlay_manifest_sha256") != sha256(manifest_path):
        errors.append("paper inference validation did not bind the manifest")
    module_path = validation.get("modeling_module_path")
    if (not isinstance(module_path, str)
            or not module_path.endswith(
                "/paper_inference_overlay/inference_model/hf_qwen3/modeling_qwen3.py")):
        errors.append("paper inference validation imported another model module")
    return errors


__all__ = [
    "EXPECTED_OVERLAY",
    "EXPECTED_SOURCE",
    "audit_paper_inference_overlay",
    "sha256",
]
