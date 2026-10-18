"""Audit the arm-specific post-Stage-1 64K/128K resource ladder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from scripts.audit_in_place_ttt_2x5090_stage1_ladder import (
        HOSTNAME, UPSTREAM_COMMIT, sha256, validate_child,
    )
except ModuleNotFoundError:  # Direct `python scripts/<name>.py` execution.
    from audit_in_place_ttt_2x5090_stage1_ladder import (
        HOSTNAME, UPSTREAM_COMMIT, sha256, validate_child,
    )


PROTOCOL = "in_place_ttt_2x5090_stage2_ladder_audit_v1"
RUN_PROTOCOL = "in_place_ttt_2x5090_stage2_ladder_v1"
ARMS = ("no_ttt", "ttt")
LENGTHS = (65536, 131072)


def validate_start_model(
    stage1_root: Path, stage1_audit: dict[str, Any], arm: str
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    saved = stage1_audit.get("arms", {}).get(arm, {})
    relative = saved.get("final_hf_relative_path")
    if not isinstance(relative, str):
        return [f"{arm}: Stage-1 audit has no HF path"], {}
    root = stage1_root / arm / relative
    manifest = saved.get("hf_manifest")
    if not root.is_dir() or not isinstance(manifest, dict) or not manifest:
        return [f"{arm}: Stage-1 HF root/manifest is missing"], {}
    current_files = {
        str(path.relative_to(root)): {
            "sha256": sha256(path), "size_bytes": path.stat().st_size,
        }
        for path in root.rglob("*") if path.is_file()
    }
    if current_files != manifest:
        errors.append(f"{arm}: Stage-1 HF manifest is stale")
    config_path = root / "config.json"
    if not config_path.is_file():
        errors.append(f"{arm}: Stage-1 HF config is missing")
        config = {}
    else:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if (bool(config.get("ttt_mode", False)) != (arm == "ttt")
                or config.get("max_position_embeddings") != 32768):
            errors.append(f"{arm}: Stage-1 HF config differs from audited arm")
    return errors, {
        "root": str(root.resolve()),
        "config_sha256": sha256(config_path) if config_path.is_file() else None,
        "manifest": current_files,
    }


def audit(
    root: Path,
    *,
    stage1_root: Path,
    resource_plan: Path,
    execution_addendum: Path,
    input_addendum: Path,
    data_manifest_path: Path,
    data_audit_path: Path,
    configs: dict[str, Path],
    runner: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    required = {name: root / name for name in (
        "ladder_metadata.json", "config_audit.json", "start_models.json", "state.txt",
    )}
    stage1_audit_path = stage1_root / "independent_audit.json"
    stage1_state_path = stage1_root / "state.txt"
    if any(not path.is_file() for path in required.values()) \
            or not stage1_audit_path.is_file() or not stage1_state_path.is_file():
        return {"protocol": PROTOCOL, "passed": False,
                "errors": ["ladder metadata/config/state or Stage-1 audit is missing"]}
    metadata = json.loads(required["ladder_metadata.json"].read_text())
    config_audit = json.loads(required["config_audit.json"].read_text())
    saved_start_models = json.loads(required["start_models.json"].read_text())
    stage1_audit = json.loads(stage1_audit_path.read_text())
    data_manifest = json.loads(data_manifest_path.read_text())
    data_audit = json.loads(data_audit_path.read_text())
    manifest_sha, input_audit_sha = sha256(data_manifest_path), sha256(data_audit_path)
    if (metadata.get("protocol") != RUN_PROTOCOL
            or metadata.get("hostname") != HOSTNAME
            or metadata.get("upstream_commit") != UPSTREAM_COMMIT
            or metadata.get("upstream_clean") is not True
            or metadata.get("tool_clean") is not True
            or not isinstance(metadata.get("tool_commit"), str)
            or len(metadata["tool_commit"]) != 40
            or metadata.get("stage1_root") != str(stage1_root.resolve())):
        errors.append("Stage-2 ladder metadata host/revision/Stage-1 binding is invalid")
    expected = {
        "resource_plan_sha256": sha256(resource_plan),
        "execution_addendum_sha256": sha256(execution_addendum),
        "input_addendum_sha256": sha256(input_addendum),
        "data_manifest_sha256": manifest_sha,
        "data_audit_sha256": input_audit_sha,
        "stage1_audit_sha256": sha256(stage1_audit_path),
        "config_audit_sha256": sha256(required["config_audit.json"]),
        "start_models_sha256": sha256(required["start_models.json"]),
        "runner_sha256": sha256(runner),
        "auditor_sha256": sha256(Path(__file__).resolve()),
        "no_ttt_config_sha256": sha256(configs["no_ttt"]),
        "ttt_config_sha256": sha256(configs["ttt"]),
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        errors.append("Stage-2 ladder dependency hash mismatch")
    if config_audit.get("passed") is not True:
        errors.append("registered 2x5090 config audit did not pass")
    if (stage1_audit.get("protocol")
            != "in_place_ttt_2x5090_100m_stage1_audit_v1"
            or stage1_audit.get("passed") is not True
            or stage1_state_path.read_text().strip() != "completed_100m_stage1"):
        errors.append("paired formal Stage-1 audit/state did not pass")
    if (data_manifest.get("protocol") != "in_place_ttt_2x5090_resource_inputs_v1"
            or data_audit.get("passed") is not True
            or data_audit.get("bindings", {}).get("manifest_sha256") != manifest_sha):
        errors.append("exact resource-input manifest/audit is invalid")
    if required["state.txt"].read_text().strip() not in {
            "all_children_completed", "completed_stage2_ladder"}:
        errors.append("Stage-2 ladder children are not complete")

    start_models: dict[str, Any] = {}
    for arm in ARMS:
        model_errors, record = validate_start_model(stage1_root, stage1_audit, arm)
        errors.extend(model_errors)
        start_models[arm] = record
    if saved_start_models != start_models:
        errors.append("saved Stage-1 start-model manifests are stale")

    boundaries: dict[str, Any] = {}
    for length in LENGTHS:
        saved_input = data_manifest.get("outputs", {}).get(str(length), {})
        audited_input = data_audit.get("outputs", {}).get(str(length), {})
        artifact = saved_input.get("artifact", {})
        data_path = data_manifest_path.parent / artifact.get("path", "")
        if (not data_path.is_file() or artifact.get("sha256") != sha256(data_path)
                or audited_input.get("artifact_sha256") != artifact.get("sha256")
                or saved_input.get("records") != 2
                or saved_input.get("training_tokens") != 2 * length):
            errors.append(f"{length}: exact input artifact is invalid")
        for arm in ARMS:
            child = root / f"resource_{length}_{arm}"
            child_errors, record = validate_child(
                child, length, arm, artifact.get("sha256", ""),
                manifest_sha, input_audit_sha,
            )
            errors.extend(child_errors)
            metadata_path = child / "run_metadata.json"
            if metadata_path.is_file():
                child_metadata = json.loads(metadata_path.read_text())
                model = start_models.get(arm, {})
                if (child_metadata.get("model") != model.get("root")
                        or child_metadata.get("model_config_sha256")
                        != model.get("config_sha256")
                        or child_metadata.get("config_sha256")
                        != sha256(configs[arm])):
                    errors.append(f"{length}/{arm}: start model/config binding differs")
            boundaries[f"{length}_{arm}"] = record
    return {
        "protocol": PROTOCOL,
        "passed": not errors,
        "errors": errors,
        "start_models": start_models,
        "boundaries": boundaries,
        "bindings": expected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage1-root", type=Path, required=True)
    parser.add_argument("--resource-plan", type=Path, required=True)
    parser.add_argument("--execution-addendum", type=Path, required=True)
    parser.add_argument("--input-addendum", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--data-audit", type=Path, required=True)
    parser.add_argument("--no-ttt-config", type=Path, required=True)
    parser.add_argument("--ttt-config", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {key: value.expanduser().resolve() for key, value in vars(args).items()
             if isinstance(value, Path)}
    output = paths.pop("output")
    configs = {"no_ttt": paths.pop("no_ttt_config"),
               "ttt": paths.pop("ttt_config")}
    paths["data_manifest_path"] = paths.pop("data_manifest")
    paths["data_audit_path"] = paths.pop("data_audit")
    root = paths.pop("root")
    result = audit(root, configs=configs, **paths)
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    partial.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    partial.replace(output)
    print(json.dumps({"output": str(output), "passed": result["passed"],
                      "errors": result["errors"]}, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
