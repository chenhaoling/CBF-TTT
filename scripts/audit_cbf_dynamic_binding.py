"""Independent artifact audit for joint CE versus dynamic binding training."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scripts.audit_cbf_dynamic_memory import audit_data, load_checkpoint
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_dynamic_binding import ARMS, schedule, summarize
from tasks.train_cbf_dynamic_memory import aggregate, select_view
from tasks.train_cbf_event_writer import write_json


def audit_arm(root: Path, rows: list[dict], arm: str) -> dict:
    import torch

    output = root / arm
    manifest = json.loads((output / "training_manifest.json").read_text())
    complete = json.loads((output / "complete.json").read_text())
    selected, groups = select_view(rows, len(manifest["groups"]["train"]), len(manifest["groups"]["dev"]))
    expected = schedule(selected, manifest["rounds"], manifest["order_seed"])
    steps = [json.loads(line) for line in (output / "steps.jsonl").read_text().splitlines() if line]
    if manifest["arm"] != arm or manifest["protocol"] != "dynamic_binding_writer_v1" or groups != manifest["groups"]:
        raise ValueError("binding manifest mismatch")
    if [row["ids"] for row in steps] != expected or any(len(row["ids"]) != 4 for row in steps):
        raise ValueError("binding schedule mismatch")
    if complete["steps_sha256"] != sha(output / "steps.jsonl") or len(steps) != complete["optimizer_updates"]:
        raise ValueError("binding step coverage mismatch")
    for row in steps:
        if not all(math.isfinite(row[key]) for key in ("loss", "balanced_loss", "contrast_loss", "max_relative_delta")):
            raise ValueError("nonfinite binding step")
        expected_loss = (sum(row["question_losses"]) / 4 if arm == "joint_ce" else
                         row["balanced_loss"] + .5 * row["contrast_loss"])
        if not math.isclose(row["loss"], expected_loss, rel_tol=1e-6, abs_tol=2e-6):
            raise ValueError("binding objective accounting mismatch")
        if (arm == "joint_ce") != (row["contrast_loss"] == 0):
            raise ValueError("binding contrast-arm mismatch")
        if row["max_relative_delta"] > 1:
            raise ValueError("binding safety invariant violated")
    initial = load_checkpoint(output / "writer_0.pt")
    final = load_checkpoint(output / f"writer_{manifest['question_exposures']}.pt")
    if initial["manifest"] != manifest or final["manifest"] != manifest:
        raise ValueError("binding checkpoint provenance mismatch")
    for name in initial["writer"]:
        delta = float((final["writer"][name] - initial["writer"][name]).norm())
        if initial["writer"][name].dtype != torch.float32 or final["writer"][name].dtype != torch.float32:
            raise ValueError("binding writer is not FP32")
        if delta <= 0 or not math.isclose(delta, complete["writer_update_norms"][name], rel_tol=2e-6, abs_tol=1e-8):
            raise ValueError("binding writer delta mismatch")
    optimizer = load_checkpoint(output / f"optimizer_{manifest['question_exposures']}.pt")
    if not all(value.dtype == torch.float32 for state in optimizer["state"].values()
               for value in state.values() if isinstance(value, torch.Tensor)):
        raise ValueError("binding optimizer is not FP32")
    evaluation_rows = 8 * 8 + len(manifest["groups"]["dev"]) * 8
    resources = []
    for exposure in manifest["checkpoints"]:
        raw_path = output / f"evaluation_{exposure}.jsonl"
        raw = [json.loads(line) for line in raw_path.read_text().splitlines() if line]
        stored = json.loads((output / f"evaluation_{exposure}.summary.json").read_text())
        if len(raw) != evaluation_rows or stored["rows_sha256"] != sha(raw_path):
            raise ValueError("binding evaluation coverage mismatch")
        if aggregate(raw) != {key: stored[key] for key in ("train_probe", "dev")}:
            raise ValueError("binding evaluation aggregation mismatch")
        for row in raw:
            resources.extend(row["policies"].values())
    return {
        "arm": arm,
        "updates": len(steps),
        "question_exposures": sum(len(batch) for batch in expected),
        "context_writes": len(steps) * manifest["context_writes_per_update"],
        "peak_training_allocated_gib": complete["peak_allocated_gib"],
        "peak_training_reserved_gib": complete["peak_reserved_gib"],
        "max_relative_delta": complete["max_relative_delta"],
        "max_evaluation_seconds": max(item["seconds"] for item in resources),
        "backbone_sha256": manifest["backbone_sha256"],
        "initial": initial,
    }


def audit(root: Path, data_root: Path) -> dict:
    rows, data_manifest, source_manifest = audit_data(data_root)
    reports = {arm: audit_arm(root, rows, arm) for arm in ARMS}
    left, right = (reports[arm]["initial"] for arm in ARMS)
    if set(left["writer"]) != set(right["writer"]) or any(
        not left["writer"][name].equal(right["writer"][name]) for name in left["writer"]
    ):
        raise ValueError("binding arms have different initial tensors")
    if reports[ARMS[0]]["backbone_sha256"] != reports[ARMS[1]]["backbone_sha256"]:
        raise ValueError("binding arms have different backbones")
    for report in reports.values():
        report.pop("initial")
    stored = json.loads((root / "summary.json").read_text())
    if summarize(root, save=False) != stored:
        raise ValueError("binding top-level summary mismatch")
    result = {
        "protocol": "dynamic_binding_writer_v1_audit",
        "passed": True,
        "data": {"rows": len(rows), "data_sha256": data_manifest["data_sha256"],
                 "backgrounds_sha256": data_manifest["backgrounds_sha256"],
                 "semantic_seed": source_manifest["seed"], "test_tokenized": False},
        "arms": reports,
        "stage_passed": stored["stage_passed"],
    }
    write_json(root / "execution_audit.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(Path(args.root), Path(args.data_root)), indent=2))


if __name__ == "__main__":
    main()
