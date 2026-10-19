#!/usr/bin/env python3
"""Independently audit the corrected Qwen3-4B-Base Level-A closure."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

try:
    from scripts.audit_in_place_ttt_level_b_configs import read_config
except ModuleNotFoundError:  # Direct script execution.
    from audit_in_place_ttt_level_b_configs import read_config


PROTOCOL = "in_place_ttt_qwen3_4b_base_level_a_audit_v1"
UPSTREAM_COMMIT = "be2324829b0e91c8fd10a74d4b43714fde6676e1"
MODEL_REVISION = "906bfd4b4dc7f14ee4320094d8b41684abff8539"
MODEL_REPOSITORY = "Qwen/Qwen3-4B-Base"
EXPECTED_TOTALS = {
    "preliminary_100m": {
        "records": 2_048,
        "training_tokens": 98_566_144,
        "synthetic_eos_tokens": 0,
    },
    "full": {
        "records": 724_800,
        "training_tokens": 34_999_369_728,
        "synthetic_eos_tokens": 0,
    },
}
ARMS = ("no_ttt", "ttt")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def check(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def verify_hash_manifest(path: Path, root: Path) -> tuple[int, list[str]]:
    errors: list[str] = []
    entries = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            digest, raw_name = line.split(maxsplit=1)
        except ValueError:
            errors.append(f"malformed hash-manifest line: {line!r}")
            continue
        candidate = Path(raw_name.lstrip(" *"))
        if not candidate.is_absolute():
            candidate = root / candidate
        if not candidate.is_file() or sha256_file(candidate) != digest:
            errors.append(f"HF hash mismatch: {candidate}")
        entries += 1
    if entries == 0:
        errors.append("empty HF hash manifest")
    return entries, errors


def audit(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    model = args.model.resolve()
    corpus = args.corpus_root.resolve()
    resource = args.resource_inputs.resolve()
    errors: list[str] = []
    checks: dict[str, bool] = {}
    required_profile = getattr(args, "required_source_profile", "full")
    if required_profile not in EXPECTED_TOTALS:
        return {"protocol": PROTOCOL, "passed": False,
                "errors": [f"unsupported source profile: {required_profile}"]}

    required = [
        args.identity, corpus / "manifest.json", args.corpus_audit,
        resource / "manifest.json", args.resource_audit,
        args.parent_plan, args.correction_addendum, args.execution_addendum,
        args.resource_input_addendum,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return {"protocol": PROTOCOL, "passed": False,
                "errors": [f"missing required files: {missing}"]}

    identity = load_json(args.identity)
    identity_files = identity.get("files", {})
    identity_ok = (
        identity.get("passed") is True
        and all(identity.get("checks", {}).values())
        and identity.get("repository") == MODEL_REPOSITORY
        and identity.get("revision") == MODEL_REVISION
    )
    for name, saved in identity_files.items():
        path = model / name
        identity_ok &= (
            path.is_file()
            and path.stat().st_size == saved.get("size")
            and sha256_file(path) == saved.get("sha256")
        )
    checks["model_identity"] = identity_ok

    corpus_manifest_path = corpus / "manifest.json"
    corpus_manifest = load_json(corpus_manifest_path)
    corpus_audit = load_json(args.corpus_audit)
    corpus_sha = sha256_file(corpus_manifest_path)
    observed_totals = corpus_manifest.get("totals", {})
    scalar_totals = {
        key: observed_totals.get(key)
        for key in ("records", "training_tokens", "synthetic_eos_tokens")
    }
    checks["source_corpus"] = (
        corpus_manifest.get("protocol") == "in_place_ttt_public_corpus_v1"
        and corpus_manifest.get("profile") == required_profile
        and scalar_totals == EXPECTED_TOTALS[required_profile]
        and isinstance(observed_totals.get("combined_curriculum_sha256"), str)
        and len(observed_totals["combined_curriculum_sha256"]) == 64
        and corpus_audit.get("passed") is True
        and all(corpus_audit.get("checks", {}).values())
        and corpus_audit.get("bindings", {}).get("manifest_sha256") == corpus_sha
        and not corpus.with_name(corpus.name + ".partial").exists()
    )

    resource_manifest_path = resource / "manifest.json"
    resource_manifest = load_json(resource_manifest_path)
    resource_audit = load_json(args.resource_audit)
    resource_sha = sha256_file(resource_manifest_path)
    resource_bindings = resource_manifest.get("bindings", {})
    resource_8k = resource_manifest.get("outputs", {}).get("8192", {})
    checks["resource_inputs"] = (
        resource_manifest.get("protocol")
        == "in_place_ttt_2x5090_resource_inputs_v1"
        and resource_bindings.get("required_source_profile") == required_profile
        and resource_bindings.get("source_manifest_sha256") == corpus_sha
        and resource_bindings.get("expected_source_manifest_sha256") == corpus_sha
        and resource_bindings.get("source_audit_sha256")
        == sha256_file(args.corpus_audit)
        and resource_bindings.get("plan_sha256")
        == sha256_file(args.resource_input_addendum)
        and resource_audit.get("passed") is True
        and all(resource_audit.get("checks", {}).values())
        and resource_audit.get("bindings", {}).get("manifest_sha256") == resource_sha
        and resource_8k.get("records") == 2
        and resource_8k.get("training_tokens") == 16_384
        and sha256_file(resource / resource_8k.get("artifact", {}).get("path", ""))
        == resource_8k.get("artifact", {}).get("sha256")
    )

    state_path = root / "state.txt"
    conversion_state = root / "conversion_state.txt"
    smoke_state = root / "converted_qwen3_4b_base_corrected_smoke_state.txt"
    checks["terminal_states"] = (
        state_path.is_file() and state_path.read_text().strip() == "completed_both_arms"
        and conversion_state.is_file()
        and conversion_state.read_text().strip() == "completed_both_arms"
        and smoke_state.is_file()
        and smoke_state.read_text().strip() == "completed_both_arms"
    )

    plan_hashes = {
        "parent_plan_sha256": sha256_file(args.parent_plan),
        "correction_addendum_sha256": sha256_file(args.correction_addendum),
        "execution_addendum_sha256": sha256_file(args.execution_addendum),
        "resource_input_addendum_sha256":
            sha256_file(args.resource_input_addendum),
    }
    starts: list[str] = []
    arm_results: dict[str, Any] = {}
    matched_bindings: list[tuple[Any, ...]] = []
    for arm in ARMS:
        arm_root = root / f"resource_8k_{arm}"
        files = {
            "metadata": arm_root / "run_metadata.json",
            "summary": arm_root / "summary.json",
            "audit": arm_root / "independent_audit.json",
            "conversion": arm_root / "hf_ckpt" / "conversion_audit.json",
            "hf_manifest": arm_root / "hf_files.sha256",
            "smoke": root / f"ruler_4k_{arm}_step1_qwen3_4b_base_corrected_smoke"
            / "summary.json",
        }
        absent = [str(path) for path in files.values() if not path.is_file()]
        if absent:
            errors.append(f"{arm} missing artifacts: {absent}")
            continue
        metadata = load_json(files["metadata"])
        summary = load_json(files["summary"])
        child_audit = load_json(files["audit"])
        conversion = load_json(files["conversion"])
        smoke = load_json(files["smoke"])
        config = read_config(Path(metadata["config"]))
        expected_ttt = arm == "ttt"
        check(
            metadata.get("upstream_commit") == UPSTREAM_COMMIT
            and metadata.get("upstream_clean") is True
            and metadata.get("hostname") == "user-Super-Server"
            and metadata.get("model") == str(model)
            and metadata.get("sequence_length") == 8192
            and metadata.get("global_batch_size") == 2
            and metadata.get("data_manifest_sha256") == resource_sha
            and metadata.get("data_audit_sha256") == sha256_file(args.resource_audit)
            and config.get("model", {}).get("foundation", {}).get("ttt_mode")
            is expected_ttt,
            f"{arm} training metadata/config mismatch", errors,
        )
        check(
            summary.get("completed") is True
            and summary.get("exit_code") == 0
            and summary.get("activation_offload_gate_passed") is True
            and all(math.isfinite(float(summary.get(key, math.nan)))
                    for key in ("last_loss", "last_gradient_norm",
                                "optimizer_step_seconds"))
            and child_audit.get("passed") is True,
            f"{arm} resource gate/audit failed", errors,
        )
        check(
            conversion.get("arm") == arm
            and bool(conversion.get("ttt_mode")) is expected_ttt
            and ((conversion.get("ttt_tensor_count", 0) > 0) is expected_ttt),
            f"{arm} conversion audit mismatch", errors,
        )
        hf_entries, hf_errors = verify_hash_manifest(
            files["hf_manifest"], arm_root / "hf_ckpt"
        )
        errors.extend(f"{arm}: {message}" for message in hf_errors)
        smoke_dependencies = smoke.get("dependencies", {})
        check(
            smoke.get("completed") is True
            and smoke.get("task_count") == 13
            and smoke.get("macro_matches_official") is True
            and smoke.get("tokenizer_regex_warning_count") == 0
            and smoke_dependencies.get("model_hash_manifest_sha256")
            == sha256_file(files["hf_manifest"])
            and smoke_dependencies.get("tokenizer_json_sha256")
            == identity_files.get("tokenizer.json", {}).get("sha256")
            and smoke_dependencies.get("model_tokenizer_json_sha256")
            == identity_files.get("tokenizer.json", {}).get("sha256")
            and smoke_dependencies.get("fix_mistral_regex") is False,
            f"{arm} RULER smoke mismatch", errors,
        )
        start_file = root / f"{arm}_started_at.txt"
        if not start_file.is_file():
            errors.append(f"{arm} start timestamp missing")
        else:
            starts.append(start_file.read_text().strip())
        matched_bindings.append((
            metadata.get("data_sha256"), metadata.get("data_manifest_sha256"),
            metadata.get("data_audit_sha256"), metadata.get("model_config_sha256"),
            metadata.get("sequence_length"), metadata.get("global_batch_size"),
        ))
        arm_results[arm] = {
            "resource_summary_sha256": sha256_file(files["summary"]),
            "resource_audit_sha256": sha256_file(files["audit"]),
            "loss": summary.get("last_loss"),
            "gradient_norm": summary.get("last_gradient_norm"),
            "minimum_gpu_free_fraction": summary.get("minimum_gpu_free_fraction"),
            "checkpoint_bytes": summary.get("checkpoint_bytes"),
            "hf_manifest_sha256": sha256_file(files["hf_manifest"]),
            "hf_manifest_entries": hf_entries,
            "ruler_summary_sha256": sha256_file(files["smoke"]),
            "ruler_macro_accuracy": smoke.get("macro_accuracy"),
            "ruler_official_score": smoke.get("official_ruler_group_score"),
        }

    checks["both_arms"] = set(arm_results) == set(ARMS)
    checks["matched_inputs"] = len(matched_bindings) == 2 \
        and len(set(matched_bindings)) == 1
    checks["execution_order"] = len(starts) == 2 and starts[0] < starts[1]
    for name, passed in checks.items():
        if not passed:
            errors.append(f"failed check: {name}")
    return {
        "protocol": PROTOCOL,
        "passed": not errors,
        "errors": errors,
        "checks": checks,
        "model_revision": MODEL_REVISION,
        "source_profile": required_profile,
        "corpus_manifest_sha256": corpus_sha,
        "resource_manifest_sha256": resource_sha,
        "plans": plan_hashes,
        "arms": arm_results,
        "claim_boundary": (
            "Corrected Qwen3-4B-Base Level-A execution closure only; one "
            "RULER example per task is not an efficacy estimate and does not "
            "release formal training, the RULER matrix, or gate exploration."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--corpus-audit", type=Path, required=True)
    parser.add_argument("--resource-inputs", type=Path, required=True)
    parser.add_argument("--resource-audit", type=Path, required=True)
    parser.add_argument("--parent-plan", type=Path, required=True)
    parser.add_argument("--correction-addendum", type=Path, required=True)
    parser.add_argument("--execution-addendum", type=Path, required=True)
    parser.add_argument("--resource-input-addendum", type=Path, required=True)
    parser.add_argument(
        "--required-source-profile",
        choices=tuple(EXPECTED_TOTALS),
        default="full",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "passed": result["passed"],
                      "errors": result["errors"]}, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
