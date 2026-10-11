"""Finalize a dynamic-memory run with a complete joint arm and safe sequential stop."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from scripts.audit_cbf_dynamic_memory import audit_arm, audit_data, load_checkpoint
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_dynamic_memory import aggregate, schedule, select_view
from tasks.train_cbf_event_writer import write_json


def audit_evaluation(output: Path, exposure: int, expected_rows: int) -> dict:
    raw_path = output / f"evaluation_{exposure}.jsonl"
    summary_path = output / f"evaluation_{exposure}.summary.json"
    raw = [json.loads(line) for line in raw_path.read_text().splitlines() if line]
    stored = json.loads(summary_path.read_text())
    if len(raw) != expected_rows or stored["rows_sha256"] != sha(raw_path):
        raise ValueError("partial-arm evaluation coverage/hash mismatch")
    if aggregate(raw) != {key: stored[key] for key in ("train_probe", "dev")}:
        raise ValueError("partial-arm evaluation aggregate mismatch")
    if stored["checkpoint_sha256"] != sha(output / f"writer_{exposure}.pt"):
        raise ValueError("partial-arm checkpoint hash mismatch")
    return stored


def finalize(root: Path, data_root: Path, run_log: Path) -> tuple[dict, dict]:
    import torch

    rows, data_manifest, source_manifest = audit_data(data_root)
    joint_report = audit_arm(root, rows, "joint_context")
    joint_initial = joint_report.pop("initial")
    joint_output = root / "joint_context"
    joint_manifest = json.loads((joint_output / "training_manifest.json").read_text())
    joint_checkpoints = {
        str(exposure): json.loads((joint_output / f"evaluation_{exposure}.summary.json").read_text())
        for exposure in joint_manifest["checkpoints"]
    }

    output = root / "sequential"
    manifest = json.loads((output / "training_manifest.json").read_text())
    selected_rows, selected_groups = select_view(rows,
        len(manifest["groups"]["train"]), len(manifest["groups"]["dev"]))
    if manifest["arm"] != "sequential" or selected_groups != manifest["groups"]:
        raise ValueError("sequential manifest mismatch")
    expected = schedule(selected_rows, "sequential", manifest["rounds"], manifest["order_seed"])
    steps = [json.loads(line) for line in (output / "steps.jsonl").read_text().splitlines() if line]
    if not manifest["checkpoints"][1] < len(steps) < manifest["checkpoints"][2]:
        raise ValueError("sequential stop is not between fixed midpoint and endpoint")
    if [row["ids"] for row in steps] != expected[:len(steps)]:
        raise ValueError("sequential completed prefix differs from preregistered schedule")
    if any(row["exposure_start"] != index or row["exposure_end"] != index + 1
           for index, row in enumerate(steps)):
        raise ValueError("sequential exposure accounting mismatch")
    if any(not math.isfinite(row["loss"]) or row["max_relative_delta"] >= 1 for row in steps):
        raise ValueError("saved sequential step violates finite/safety invariant")
    log_text = run_log.read_text()
    if "dynamic writer delta exceeds ||delta||/||W|| safety tripwire" not in log_text:
        raise ValueError("sequential safety-stop evidence missing from run log")
    if (output / "complete.json").exists() or (output / "evaluation_2048.summary.json").exists():
        raise ValueError("sequential arm unexpectedly claims endpoint completion")

    expected_rows = 8 * 8 + len(manifest["groups"]["dev"]) * 8
    sequential_checkpoints = {
        str(exposure): audit_evaluation(output, exposure, expected_rows)
        for exposure in manifest["checkpoints"][:2]
    }
    initial = load_checkpoint(output / "writer_0.pt")
    midpoint = load_checkpoint(output / f"writer_{manifest['checkpoints'][1]}.pt")
    if set(initial["writer"]) != set(joint_initial["writer"]) or any(
        not initial["writer"][name].equal(joint_initial["writer"][name]) for name in initial["writer"]
    ):
        raise ValueError("joint/sequential initial writer tensors differ")
    if any(value.dtype != torch.float32 for state in (initial, midpoint)
           for value in state["writer"].values()):
        raise ValueError("partial writer checkpoint is not FP32")
    if any(float((midpoint["writer"][name] - initial["writer"][name]).norm()) <= 0
           for name in initial["writer"]):
        raise ValueError("sequential midpoint writer did not update")
    midpoint_key = str(manifest["checkpoints"][1])
    for split in ("train_probe", "dev"):
        if sequential_checkpoints["0"][split]["means"]["empty"] != sequential_checkpoints[midpoint_key][split]["means"]["empty"]:
            raise ValueError("sequential frozen empty control drift")

    joint_final = joint_checkpoints[str(joint_manifest["question_exposures"])]
    summary = {
        "protocol": "dynamic_memory_writer_v1_safety_stop",
        "data_sha256": data_manifest["data_sha256"],
        "semantic_seed": source_manifest["seed"],
        "arms": {
            "joint_context": {
                "status": "complete", "manifest": joint_manifest,
                "complete": json.loads((joint_output / "complete.json").read_text()),
                "checkpoints": joint_checkpoints,
                "final_gate": joint_final["content_gate"],
            },
            "sequential": {
                "status": "stopped_by_delta_safety_tripwire",
                "manifest": manifest,
                "completed_updates": len(steps),
                "completed_question_exposures": steps[-1]["exposure_end"],
                "failed_update": len(steps) + 1,
                "checkpoints": sequential_checkpoints,
                "max_completed_relative_delta": max(row["max_relative_delta"] for row in steps),
                "last50_mean_loss": statistics.fmean(row["loss"] for row in steps[-50:]),
            },
        },
        "stage_passed": False,
        "stage_reason": "joint endpoint failed the fixed content gate; sequential stopped before endpoint at the delta safety tripwire",
        "test_scored": False,
        "terminal_status": "completed_with_sequential_safety_stop",
    }
    write_json(root / "summary.json", summary)
    audit = {
        "protocol": "dynamic_memory_writer_v1_safety_stop_audit",
        "passed": True,
        "data": {"rows": len(rows), "data_sha256": data_manifest["data_sha256"],
                 "backgrounds_sha256": data_manifest["backgrounds_sha256"], "test_tokenized": False},
        "joint_context": joint_report,
        "sequential": {
            "completed_updates": len(steps), "failed_update": len(steps) + 1,
            "steps_sha256": sha(output / "steps.jsonl"),
            "run_log_sha256": sha(run_log),
            "evaluated_checkpoints": list(sequential_checkpoints),
            "schedule_prefix_verified": True, "initial_tensors_match_joint": True,
        },
        "stage_passed": False,
    }
    write_json(root / "execution_audit.json", audit)
    return summary, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--run-log", required=True)
    args = parser.parse_args()
    summary, audit = finalize(Path(args.root), Path(args.data_root), Path(args.run_log))
    print(json.dumps({"summary": summary, "audit": audit}, indent=2))


if __name__ == "__main__":
    main()
