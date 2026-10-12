"""Independent artifact audit for final-checkpoint writer binding recovery."""

import argparse
import json
from pathlib import Path

from tasks.build_cbf_memory_value import file_digest
from tasks.train_final_writer_binding import (TUNE_ARMS, load, summarize_files)


def audit(root):
    root = Path(root)
    rows, design = load(root)
    checks = {"expanded_rows": len(rows) == 192,
              "balanced_groups": all(sum(row["group"] == group for row in rows) == 8 for group in range(24)),
              "source_locked": all(row["group"] < 24 for row in rows)}
    initial = set()
    for arm, lr in TUNE_ARMS.items():
        folder = root / arm
        manifest = json.loads((folder / "training_manifest.json").read_text())
        complete = json.loads((folder / "training_complete.json").read_text())
        initial.add(manifest["initial_writer_sha256"])
        checks[f"{arm}_budget"] = manifest["updates"] == 192 and complete["updates"] == 192
        checks[f"{arm}_lr"] = manifest["lr"] == lr
        checks[f"{arm}_backbone"] = complete["backbone_unchanged"] and complete["optimizer_state_fp32"]
        checks[f"{arm}_hashes"] = (complete["steps_sha256"] == file_digest(folder / "steps.jsonl") and
                                    complete["checkpoint_sha256"] == file_digest(folder / "writer.pt"))
        rebuilt = summarize_files(root, [folder / "dev.jsonl"], "dev")
        saved = json.loads((folder / "dev_summary.json").read_text())
        checks[f"{arm}_dev_recomputed"] = rebuilt == saved
    checks["identical_tune_initialization"] = len(initial) == 1
    selection = json.loads((root / "tune_selection.json").read_text())
    checks["selection_valid"] = (selection["selected_arm"] in TUNE_ARMS and
                                  selection["selected_lr"] == TUNE_ARMS[selection["selected_arm"]] and
                                  not selection["q_scored"] and not selection["v_scored"])
    final = root / "final"
    manifest = json.loads((final / "training_manifest.json").read_text())
    complete = json.loads((final / "training_complete.json").read_text())
    checks["final_budget"] = manifest["updates"] == 256 and complete["updates"] == 256
    checks["final_fresh_initialization"] = manifest["initial_writer_sha256"] in initial
    checks["final_selected_lr"] = manifest["lr"] == selection["selected_lr"]
    checks["final_hashes"] = (complete["steps_sha256"] == file_digest(final / "steps.jsonl") and
                               complete["checkpoint_sha256"] == file_digest(final / "writer.pt"))
    q_paths = [final / "q_0.jsonl", final / "q_1.jsonl"]
    q = summarize_files(root, q_paths, "q")
    saved_q = json.loads((final / "q_summary.json").read_text())
    checks["q_recomputed"] = q == saved_q
    v_paths = [final / "v_0.jsonl", final / "v_1.jsonl"]
    v_exists = all(path.exists() for path in v_paths)
    checks["v_lock_respected"] = v_exists == bool(q["passed"])
    saved_v = None
    if v_exists:
        saved_v = json.loads((final / "v_summary.json").read_text())
        checks["v_recomputed"] = summarize_files(root, v_paths, "v") == saved_v
    checks["all_norm_matched"] = all(
        abs(value - 1) <= 1e-6 for path in q_paths + (v_paths if v_exists else [])
        for row in map(json.loads, path.read_text().splitlines())
        for value in row["random_norm_ratios"].values())
    result = {"protocol": "final_writer_binding_v1", "audit_passed": all(checks.values()),
              "checks": checks, "q_passed": q["passed"], "v_scored": v_exists,
              "decision": saved_v if saved_v is not None else q,
              "source_data_sha256": design["source_data_sha256"],
              "q_summary_sha256": file_digest(final / "q_summary.json"),
              "v_summary_sha256": file_digest(final / "v_summary.json") if v_exists else None}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(); result = audit(args.root)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["audit_passed"]: raise SystemExit(1)


if __name__ == "__main__": main()
