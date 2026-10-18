#!/usr/bin/env python3
"""Project the corrected Base matched pair from audited 32K/128K timings."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


PROTOCOL = "in_place_ttt_qwen3_4b_base_scale_feasibility_v1"
MODEL_REVISION = "906bfd4b4dc7f14ee4320094d8b41684abff8539"
ARMS = ("no_ttt", "ttt")
STAGES = {
    "stage1": {
        "protocol": "in_place_ttt_qwen3_4b_base_stage1_timing_audit_v1",
        "selected_tokens_per_arm": 56_623_104,
        "paper_tokens_per_arm": 20_000_000_000,
        "projection_key": "projected_sequential_stage1_seconds",
    },
    "stage2": {
        "protocol": "in_place_ttt_qwen3_4b_base_stage2_timing_audit_v1",
        "selected_tokens_per_arm": 41_943_040,
        "paper_tokens_per_arm": 15_000_000_000,
        "projection_key": "projected_sequential_stage2_seconds",
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(stage1_audit: Path, stage2_audit: Path, plan: Path) -> dict[str, Any]:
    paths = {"stage1": stage1_audit, "stage2": stage2_audit}
    stages: dict[str, Any] = {}
    total_seconds = 0.0
    for name, spec in STAGES.items():
        payload = json.loads(paths[name].read_text(encoding="utf-8"))
        if (payload.get("protocol") != spec["protocol"]
                or payload.get("passed") is not True
                or payload.get("errors") != []
                or payload.get("model_revision") != MODEL_REVISION
                or set(payload.get("arms", {})) != set(ARMS)):
            raise ValueError(f"{name}: corrected timing audit is invalid")
        arms: dict[str, Any] = {}
        selected_pair = 0.0
        paper_pair = 0.0
        for arm in ARMS:
            record = payload["arms"][arm]
            throughput = record.get("steady_tokens_per_second")
            measured = record.get(f"projected_formal_{name}_seconds")
            if (not isinstance(throughput, (int, float))
                    or isinstance(throughput, bool) or not math.isfinite(throughput)
                    or throughput <= 0 or not isinstance(measured, (int, float))
                    or isinstance(measured, bool) or not math.isfinite(measured)
                    or measured <= 0):
                raise ValueError(f"{name}/{arm}: timing is invalid")
            expected_selected = spec["selected_tokens_per_arm"] / throughput
            if abs(float(measured) - expected_selected) >= 1e-9 * expected_selected:
                raise ValueError(f"{name}/{arm}: selected-tier projection differs")
            paper_seconds = spec["paper_tokens_per_arm"] / throughput
            selected_pair += expected_selected
            paper_pair += paper_seconds
            arms[arm] = {
                "steady_tokens_per_second": float(throughput),
                "selected_tier_tokens": spec["selected_tokens_per_arm"],
                "selected_tier_seconds": expected_selected,
                "paper_tokens": spec["paper_tokens_per_arm"],
                "paper_seconds": paper_seconds,
                "paper_hours": paper_seconds / 3600,
                "paper_days": paper_seconds / 86400,
                "gpu": record.get("gpu"),
                "host": record.get("host"),
            }
        registered_pair = payload.get(spec["projection_key"])
        if (not isinstance(registered_pair, (int, float))
                or abs(float(registered_pair) - selected_pair)
                >= 1e-9 * selected_pair):
            raise ValueError(f"{name}: paired selected-tier projection differs")
        stages[name] = {
            "sequence_length": 32768 if name == "stage1" else 131072,
            "arms": arms,
            "selected_pair_tokens": 2 * spec["selected_tokens_per_arm"],
            "selected_pair_seconds": selected_pair,
            "paper_pair_tokens": 2 * spec["paper_tokens_per_arm"],
            "paper_pair_seconds": paper_pair,
            "paper_pair_hours": paper_pair / 3600,
            "paper_pair_days": paper_pair / 86400,
        }
        total_seconds += paper_pair
    return {
        "protocol": PROTOCOL,
        "completed": True,
        "model_revision": MODEL_REVISION,
        "execution_host": "hku-gpu2",
        "gpu_count": 2,
        "gpu_model": "RTX 5090",
        "arms_run_sequentially": True,
        "tokens_per_model": 35_000_000_000,
        "matched_pair_processed_tokens": 70_000_000_000,
        "stages": stages,
        "paper_pair_total_seconds": total_seconds,
        "paper_pair_total_hours": total_seconds / 3600,
        "paper_pair_total_days": total_seconds / 86400,
        "within_30_days": total_seconds <= 30 * 86400,
        "within_90_days": total_seconds <= 90 * 86400,
        "speedup_required_for_30_days": total_seconds / (30 * 86400),
        "speedup_required_for_90_days": total_seconds / (90 * 86400),
        "bindings": {
            "corrected_stage1_timing_audit_sha256": sha256(stage1_audit),
            "corrected_stage2_timing_audit_sha256": sha256(stage2_audit),
            "plan_sha256": sha256(plan),
            "reporter_sha256": sha256(Path(__file__).resolve()),
        },
        "paper_scale_executed": False,
        "projection_only": True,
        "gate_experiment_started": False,
        "claim_boundary": (
            "measured_short_run_linear_time_projection_not_paper_scale_execution"
        ),
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Qwen3-4B-Base scale feasibility on 2xRTX5090", "",
        "This is a linear projection from audited three-step steady-state "
        "timings. It is not a completed paper-scale run.", "",
        "| Stage | Arm | Tokens/s | Target tokens | Hours | Days |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for stage in ("stage1", "stage2"):
        for arm in ARMS:
            row = result["stages"][stage]["arms"][arm]
            lines.append(
                f"| {stage} | {arm} | {row['steady_tokens_per_second']:.2f} | "
                f"{row['paper_tokens']:,} | {row['paper_hours']:.2f} | "
                f"{row['paper_days']:.2f} |"
            )
    lines += [
        "", "## Sequential matched-pair projection", "",
        f"Both arms process **{result['matched_pair_processed_tokens']:,} tokens** "
        f"in total. Projected wall time is **{result['paper_pair_total_days']:.2f} "
        "days** because each measured arm occupies both GPUs.", "",
        f"Required speedup for 30 days: **{result['speedup_required_for_30_days']:.2f}x**. "
        f"Required speedup for 90 days: **{result['speedup_required_for_90_days']:.2f}x**.",
        "", "`paper_scale_executed=false`; `projection_only=true`; "
        "`gate_experiment_started=false`.", "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-audit", type=Path, required=True)
    parser.add_argument("--stage2-audit", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise RuntimeError("scale feasibility reporter must hide all GPUs")
    stage1, stage2, plan = (
        args.stage1_audit.resolve(), args.stage2_audit.resolve(), args.plan.resolve()
    )
    outputs = (args.output_json.resolve(), args.output_md.resolve())
    for path in outputs:
        if path.exists() or path.with_name(path.name + ".partial").exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    result = summarize(stage1, stage2, plan)
    outputs[0].parent.mkdir(parents=True, exist_ok=True)
    json_partial = outputs[0].with_name(outputs[0].name + ".partial")
    md_partial = outputs[1].with_name(outputs[1].name + ".partial")
    json_partial.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    md_partial.write_text(render_markdown(result))
    os.replace(json_partial, outputs[0]); os.replace(md_partial, outputs[1])
    print(json.dumps({"completed": True, "json": str(outputs[0]),
                      "markdown": str(outputs[1])}, sort_keys=True))


if __name__ == "__main__":
    main()
