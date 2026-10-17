#!/usr/bin/env python3
"""Independently audit the completed exclusive-5090 baseline resource run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())


def gpu_peaks(path: pathlib.Path) -> tuple[dict[str, float], dict[str, float]]:
    peaks: dict[str, float] = {}
    totals: dict[str, float] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            device = row["index"]
            used = float(row["memory_used_mib"])
            total = float(row["memory_total_mib"])
            peaks[device] = max(peaks.get(device, 0.0), used)
            if device in totals and totals[device] != total:
                raise ValueError(f"GPU{device} total memory changed")
            totals[device] = total
    if not peaks:
        raise ValueError(f"empty GPU trace: {path}")
    return peaks, totals


def audit(root: pathlib.Path) -> dict:
    state = (root / "state.txt").read_text().strip()
    conversion_state = (root / "conversion_state.txt").read_text().strip()
    smoke_state = (root / "converted_smoke_state.txt").read_text().strip()
    if state != "completed_both_arms":
        raise ValueError(f"unexpected training state: {state}")
    if conversion_state != "completed_both_arms":
        raise ValueError(f"unexpected conversion state: {conversion_state}")
    if smoke_state != "completed_both_arms":
        raise ValueError(f"unexpected smoke state: {smoke_state}")

    training = {}
    conversion = {}
    ruler = {}
    input_bindings = []
    minimum_free_fractions = []
    for arm in ("no_ttt", "ttt"):
        arm_root = root / f"resource_8k_{arm}"
        summary = load_json(arm_root / "summary.json")
        if summary.get("exit_code") != 0 or not summary.get("completed"):
            raise ValueError(f"{arm} training did not complete")
        peaks, totals = gpu_peaks(arm_root / "gpu.csv")
        recorded = {
            str(device): float(value)
            for device, value in summary["peak_gpu_memory_mib"].items()
        }
        if peaks != recorded:
            raise ValueError(f"{arm} recorded GPU peaks do not match raw trace")
        free_fractions = {
            device: (totals[device] - peak) / totals[device]
            for device, peak in peaks.items()
        }
        minimum_free_fractions.extend(free_fractions.values())
        training[arm] = {
            "summary_sha256": sha256(arm_root / "summary.json"),
            "exit_code": summary["exit_code"],
            "loss": summary["last_loss"],
            "gradient_norm": summary["last_gradient_norm"],
            "wall_seconds": summary["wall_seconds"],
            "tokens_per_second": summary["tokens_per_second"],
            "peak_gpu_memory_mib": peaks,
            "gpu_total_memory_mib": totals,
            "free_fraction_at_peak": free_fractions,
            "checkpoint_bytes": summary["checkpoint_bytes"],
        }
        deps = summary["dependencies"]
        input_bindings.append(
            (
                deps["plan_sha256"],
                deps["data_sha256"],
                deps["model_config_sha256"],
                summary["sequence_length"],
                summary["global_batch_size"],
            )
        )

        conversion_audit_path = arm_root / "hf_ckpt" / "conversion_audit.json"
        conversion_audit = load_json(conversion_audit_path)
        if conversion_audit.get("arm") != arm:
            raise ValueError(f"{arm} conversion arm mismatch")
        expected_ttt_mode = arm == "ttt"
        if bool(conversion_audit.get("ttt_mode")) != expected_ttt_mode:
            raise ValueError(f"{arm} converted config mode mismatch")
        ttt_count = int(conversion_audit.get("ttt_tensor_count", -1))
        if (arm == "ttt" and ttt_count <= 0) or (arm == "no_ttt" and ttt_count != 0):
            raise ValueError(f"{arm} native writer tensor audit failed")
        hash_manifest = arm_root / "hf_files.sha256"
        conversion[arm] = {
            **conversion_audit,
            "conversion_audit_sha256": sha256(conversion_audit_path),
            "hf_hash_manifest_sha256": sha256(hash_manifest),
            "hf_hash_manifest_entries": len(hash_manifest.read_text().splitlines()),
        }

        ruler_root = root / f"ruler_4k_{arm}_step1_smoke"
        ruler_summary_path = ruler_root / "summary.json"
        ruler_summary = load_json(ruler_summary_path)
        if not ruler_summary.get("completed") or ruler_summary.get("task_count") != 13:
            raise ValueError(f"{arm} RULER smoke incomplete")
        if not ruler_summary.get("macro_matches_official"):
            raise ValueError(f"{arm} RULER score mismatch")
        bound_hash = ruler_summary["dependencies"].get("model_hash_manifest_sha256")
        if bound_hash != conversion[arm]["hf_hash_manifest_sha256"]:
            raise ValueError(f"{arm} RULER run is not bound to converted weights")
        ruler[arm] = {
            "summary_sha256": sha256(ruler_summary_path),
            "completed": True,
            "task_count": ruler_summary["task_count"],
            "macro_accuracy": ruler_summary["macro_accuracy"],
            "official_ruler_group_score": ruler_summary["official_ruler_group_score"],
            "peak_gpu_memory_mib": ruler_summary["peak_gpu_memory_mib"],
            "tokenizer_regex_warning_count": (
                ruler_root / "eval.log"
            ).read_text(errors="replace").count("incorrect regex pattern"),
        }

    if len(set(input_bindings)) != 1:
        raise ValueError("matched arms do not share frozen input bindings")

    integrity_lines = (root / "hf_integrity_check.txt").read_text().splitlines()
    ok_lines = [line for line in integrity_lines if line.endswith(": OK")]
    if any("FAILED" in line for line in integrity_lines) or len(ok_lines) != 30:
        raise ValueError("HF checkpoint integrity check is incomplete")

    min_free = min(minimum_free_fractions)
    payload = {
        "protocol": "in_place_ttt_baseline_exclusive_independent_audit_v1",
        "execution_passed": True,
        "matched_input_bindings": True,
        "training": training,
        "conversion": conversion,
        "ruler_4k_smoke": ruler,
        "hf_integrity_ok_files": len(ok_lines),
        "minimum_gpu_free_fraction_at_peak": min_free,
        "resource_ladder_minimum_free_fraction": 0.05,
        "resource_ladder_stop_required": min_free < 0.05,
        "gradient_norm_ratio_ttt_over_no_ttt": (
            training["ttt"]["gradient_norm"] / training["no_ttt"]["gradient_norm"]
        ),
        "throughput_ratio_ttt_over_no_ttt": (
            training["ttt"]["tokens_per_second"]
            / training["no_ttt"]["tokens_per_second"]
        ),
        "claim_boundary": (
            "Level-A execution smoke only; one example per RULER task is not an "
            "efficacy estimate and 8K training peak requires stopping the 2x5090 "
            "length ladder before 16K."
        ),
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    payload = audit(args.root)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
