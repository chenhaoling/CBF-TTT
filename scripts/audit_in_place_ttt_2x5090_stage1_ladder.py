"""Independently audit the 8K prerequisite and 16K/32K hku-gpu2 ladder."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


PROTOCOL = "in_place_ttt_2x5090_stage1_ladder_audit_v1"
PREREQUISITE_PROTOCOL = "in_place_ttt_2x5090_8k_prerequisite_audit_v1"
UPSTREAM_COMMIT = "be2324829b0e91c8fd10a74d4b43714fde6676e1"
HOSTNAME = "user-Super-Server"
EIGHT_K_DATA_SHA256 = "2eb03326ec87bc831ee1441e812bb5bef843f8e906656787cd2ddafc69d9d823"
ARMS = ("no_ttt", "ttt")
LENGTHS = (16384, 32768)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def validate_child(
    root: Path,
    length: int,
    arm: str,
    expected_data_sha256: str,
    data_manifest_sha256: str | None = None,
    data_audit_sha256: str | None = None,
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    required = {
        name: root / name for name in (
            "run_metadata.json", "summary.json", "independent_audit.json",
            "train.log", "gpu.csv", "host.csv", "exit_code.txt",
        )
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        return [f"{root}: missing {missing}"], {}
    metadata = json.loads(required["run_metadata.json"].read_text())
    summary = json.loads(required["summary.json"].read_text())
    independent = json.loads(required["independent_audit.json"].read_text())
    if (metadata.get("protocol") != "in_place_ttt_baseline_resource_v2"
            or metadata.get("hostname") != HOSTNAME
            or metadata.get("upstream_commit") != UPSTREAM_COMMIT
            or metadata.get("upstream_clean") is not True
            or metadata.get("resource_profile") != "activation_offload"
            or metadata.get("activation_offload") is not True
            or metadata.get("sequence_length") != length
            or metadata.get("global_batch_size") != 2
            or metadata.get("arm") != arm
            or metadata.get("data_sha256") != expected_data_sha256):
        errors.append(f"{root}: run metadata differs from registered boundary")
    if data_manifest_sha256 is not None:
        if (metadata.get("data_manifest_sha256") != data_manifest_sha256
                or metadata.get("data_audit_sha256") != data_audit_sha256
                or summary.get("dependencies", {}).get("data_manifest_sha256")
                != data_manifest_sha256
                or summary.get("dependencies", {}).get("data_audit_sha256")
                != data_audit_sha256):
            errors.append(f"{root}: exact-input bindings differ")
    if (summary.get("completed") is not True
            or summary.get("activation_offload_gate_passed") is not True
            or summary.get("exit_code") != 0
            or summary.get("sequence_length") != length
            or summary.get("arm") != arm):
        errors.append(f"{root}: resource summary did not pass")
    for key in ("last_loss", "last_gradient_norm", "optimizer_step_seconds"):
        value = summary.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            errors.append(f"{root}: {key} is nonfinite")
    free = summary.get("minimum_gpu_free_fraction", {})
    if set(free) != {"0", "1"} or any(float(value) < 0.05 for value in free.values()):
        errors.append(f"{root}: GPU headroom gate is invalid")
    host = summary.get("host_resources") or {}
    if (host.get("minimum_memory_available_mib", 0) < 8192
            or host.get("peak_swap_used_mib", math.inf) > 3584):
        errors.append(f"{root}: host resource gate is invalid")
    if (independent.get("protocol") != "in_place_ttt_2x5090_resource_audit_v1"
            or independent.get("passed") is not True
            or independent.get("sequence_length") != length
            or independent.get("arm") != arm):
        errors.append(f"{root}: per-boundary independent audit did not pass")
    dependencies = summary.get("dependencies", {})
    current = {
        "train_log_sha256": sha256(required["train.log"]),
        "gpu_monitor_sha256": sha256(required["gpu.csv"]),
        "host_monitor_sha256": sha256(required["host.csv"]),
    }
    if any(dependencies.get(key) != digest for key, digest in current.items()):
        errors.append(f"{root}: resource summary is stale")
    checkpoint = root / "checkpoint"
    checkpoint_bytes = sum(
        path.stat().st_size for path in checkpoint.rglob("*") if path.is_file()
    ) if checkpoint.is_dir() else 0
    if checkpoint_bytes < 40_000_000_000 \
            or summary.get("checkpoint_bytes") != checkpoint_bytes:
        errors.append(f"{root}: DCP checkpoint is absent, small or stale")
    return errors, {
        "root": str(root.resolve()),
        "metadata_sha256": sha256(required["run_metadata.json"]),
        "summary_sha256": sha256(required["summary.json"]),
        "independent_audit_sha256": sha256(required["independent_audit.json"]),
        "last_loss": summary.get("last_loss"),
        "last_gradient_norm": summary.get("last_gradient_norm"),
        "optimizer_step_seconds": summary.get("optimizer_step_seconds"),
        "steady_tokens_per_second": summary.get("steady_tokens_per_second"),
        "peak_gpu_memory_mib": summary.get("peak_gpu_memory_mib"),
        "minimum_gpu_free_fraction": free,
        "host_resources": host,
        "checkpoint_bytes": checkpoint_bytes,
    }


def audit_prerequisite(root: Path) -> dict[str, Any]:
    errors: list[str] = []
    records = {}
    state_path = root / "state.txt"
    if not state_path.is_file() or state_path.read_text().strip() != "completed_8k_both_arms":
        errors.append("8K watcher has not completed both arms")
    for arm in ARMS:
        child = root / f"resource_8k_{arm}"
        child_errors, record = validate_child(
            child, 8192, arm, EIGHT_K_DATA_SHA256
        )
        errors.extend(child_errors)
        records[arm] = record
    return {
        "protocol": PREREQUISITE_PROTOCOL,
        "passed": not errors,
        "errors": errors,
        "eight_k_root": str(root.resolve()),
        "state_sha256": sha256(state_path) if state_path.is_file() else None,
        "arms": records,
    }


def audit_final(
    root: Path,
    eight_k_root: Path,
    data_manifest_path: Path,
    data_audit_path: Path,
    execution_addendum: Path,
    input_addendum: Path,
    runner: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    required = {name: root / name for name in (
        "ladder_metadata.json", "eight_k_prerequisite_audit.json",
        "config_audit.json", "state.txt",
    )}
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        return {"protocol": PROTOCOL, "passed": False,
                "errors": [f"missing ladder artifacts: {missing}"]}
    metadata = json.loads(required["ladder_metadata.json"].read_text())
    saved_prerequisite = json.loads(required["eight_k_prerequisite_audit.json"].read_text())
    current_prerequisite = audit_prerequisite(eight_k_root)
    config_audit = json.loads(required["config_audit.json"].read_text())
    data_manifest = json.loads(data_manifest_path.read_text())
    data_audit = json.loads(data_audit_path.read_text())
    manifest_sha, input_audit_sha = sha256(data_manifest_path), sha256(data_audit_path)
    if (metadata.get("protocol") != "in_place_ttt_2x5090_stage1_ladder_v1"
            or metadata.get("hostname") != HOSTNAME
            or metadata.get("upstream_commit") != UPSTREAM_COMMIT
            or metadata.get("upstream_clean") is not True
            or metadata.get("tool_clean") is not True
            or not isinstance(metadata.get("tool_commit"), str)
            or len(metadata["tool_commit"]) != 40
            or metadata.get("eight_k_root") != str(eight_k_root.resolve())):
        errors.append("ladder metadata host/upstream binding is invalid")
    expected_hashes = {
        "execution_addendum_sha256": sha256(execution_addendum),
        "input_addendum_sha256": sha256(input_addendum),
        "runner_sha256": sha256(runner),
        "auditor_sha256": sha256(Path(__file__).resolve()),
        "data_manifest_sha256": manifest_sha,
        "data_audit_sha256": input_audit_sha,
        "config_audit_sha256": sha256(required["config_audit.json"]),
        "eight_k_prerequisite_audit_sha256":
            sha256(required["eight_k_prerequisite_audit.json"]),
    }
    if any(metadata.get(key) != value for key, value in expected_hashes.items()):
        errors.append("ladder dependency hash mismatch")
    if (saved_prerequisite.get("passed") is not True
            or current_prerequisite.get("passed") is not True
            or saved_prerequisite != current_prerequisite):
        errors.append("8K prerequisite is absent, failed or stale")
    if config_audit.get("passed") is not True:
        errors.append("2x5090 config audit did not pass")
    if (data_manifest.get("protocol") != "in_place_ttt_2x5090_resource_inputs_v1"
            or data_audit.get("passed") is not True
            or data_audit.get("bindings", {}).get("manifest_sha256") != manifest_sha):
        errors.append("exact resource input audit is invalid")
    if required["state.txt"].read_text().strip() not in {
            "all_children_completed", "completed_stage1_ladder"}:
        errors.append("ladder children are not complete")

    records = {}
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
            records[f"{length}_{arm}"] = record
    return {
        "protocol": PROTOCOL,
        "passed": not errors,
        "errors": errors,
        "eight_k_prerequisite": current_prerequisite,
        "boundaries": records,
        "bindings": expected_hashes,
    }


def write_result(result: dict[str, Any], output: Path) -> None:
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    partial.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    partial.replace(output)
    print(json.dumps({"output": str(output), "passed": result["passed"],
                      "errors": result["errors"]}, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prerequisite = subparsers.add_parser("prerequisite")
    prerequisite.add_argument("--eight-k-root", type=Path, required=True)
    prerequisite.add_argument("--output", type=Path, required=True)
    final = subparsers.add_parser("final")
    final.add_argument("--root", type=Path, required=True)
    final.add_argument("--eight-k-root", type=Path, required=True)
    final.add_argument("--data-manifest", type=Path, required=True)
    final.add_argument("--data-audit", type=Path, required=True)
    final.add_argument("--execution-addendum", type=Path, required=True)
    final.add_argument("--input-addendum", type=Path, required=True)
    final.add_argument("--runner", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prerequisite":
        result = audit_prerequisite(args.eight_k_root.expanduser().resolve())
    else:
        result = audit_final(
            args.root.expanduser().resolve(),
            args.eight_k_root.expanduser().resolve(),
            args.data_manifest.expanduser().resolve(),
            args.data_audit.expanduser().resolve(),
            args.execution_addendum.expanduser().resolve(),
            args.input_addendum.expanduser().resolve(),
            args.runner.expanduser().resolve(),
        )
    write_result(result, args.output.expanduser().resolve())


if __name__ == "__main__":
    main()
