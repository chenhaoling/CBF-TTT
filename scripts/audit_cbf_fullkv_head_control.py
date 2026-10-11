"""Independent artifact audit for the direct Full-KV head control."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_fullkv_head_control import ARMS, aggregate, gate
from tasks.train_cbf_retrieval_head import head_schedule


def audit(root: str | Path, data_root: str | Path) -> dict:
    import torch

    root, data_root = Path(root), Path(data_root)
    source = json.loads((data_root / "manifest.json").read_text())
    rows_path = data_root / "rows.jsonl"
    if source["test_tokenized"] or sha(rows_path) != source["data_sha256"]:
        raise ValueError("fullkv control source seal failed")
    audits, initial = {}, None
    for arm, learning_rate in ARMS.items():
        output = root / arm
        manifest = json.loads((output / "training_manifest.json").read_text())
        complete = json.loads((output / "complete.json").read_text())
        if (manifest["protocol"] != "fullkv_head_control_v1" or manifest["arm"] != arm
                or manifest["head_lr"] != learning_rate or not manifest["writer_frozen"]
                or manifest["data_sha256"] != source["data_sha256"] or manifest["test_scored"]
                or manifest["projection_size"] != 256 or manifest["temperature"] != .1
                or sha(output / "features.pt") != manifest["feature_artifact"]["sha256"]):
            raise ValueError("fullkv control manifest mismatch")
        steps_path = output / "steps.jsonl"
        steps = [json.loads(line) for line in steps_path.read_text().splitlines() if line]
        all_rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line]
        selected_groups = manifest["groups"]
        contexts = {row["context_id"] for row in all_rows
                    if row["group_id"] in set(selected_groups["train"])}
        schedule_rows = [row for row in all_rows if row["context_id"] in contexts]
        expected = head_schedule(schedule_rows, manifest["cycles"], manifest["order_seed"])
        if (len(steps) != manifest["optimizer_updates"] or
                [(row["context_id"], *row["assignment_shifts"]) for row in steps] != expected
                or sha(steps_path) != complete["steps_sha256"]):
            raise ValueError("fullkv control schedule mismatch")
        if any(not 0 <= row["accuracy"] <= 1 or
               not all(math.isfinite(row[key]) for key in ("loss", "accuracy", "head_gradient_norm"))
               for row in steps):
            raise ValueError("invalid fullkv control step")
        for key, subset, metric in (("first32_loss", steps[:32], "loss"),
                                    ("last32_loss", steps[-32:], "loss"),
                                    ("first32_accuracy", steps[:32], "accuracy"),
                                    ("last32_accuracy", steps[-32:], "accuracy")):
            if complete[key] != statistics.fmean(row[metric] for row in subset):
                raise ValueError("fullkv control aggregate mismatch")
        zero = torch.load(output / "head_0.pt", map_location="cpu", weights_only=True)
        final = torch.load(output / f"head_{manifest['question_exposures']}.pt",
                           map_location="cpu", weights_only=True)
        if zero["manifest"] != manifest or final["manifest"] != manifest:
            raise ValueError("fullkv head checkpoint manifest mismatch")
        if initial is None:
            initial = zero["head"]
        elif any(not torch.equal(initial[name], zero["head"][name]) for name in initial):
            raise ValueError("fullkv arms do not share head initialization")
        if not all(not torch.equal(zero["head"][name], final["head"][name]) for name in initial):
            raise ValueError("fullkv head did not update")
        summaries = {}
        for exposure in (0, manifest["question_exposures"]):
            raw = output / f"evaluation_{exposure}.jsonl"
            records = [json.loads(line) for line in raw.read_text().splitlines() if line]
            stored = json.loads((output / f"evaluation_{exposure}.summary.json").read_text())
            if len(records) != 256 or stored["rows_sha256"] != sha(raw) or stored["test_scored"]:
                raise ValueError("fullkv evaluation coverage mismatch")
            if aggregate(records) != {key: stored[key] for key in ("train_probe", "dev")}:
                raise ValueError("fullkv evaluation aggregate mismatch")
            summaries[str(exposure)] = stored
        final_summary = summaries[str(manifest["question_exposures"])]
        expected_gate = gate(final_summary, complete["last32_accuracy"])
        if final_summary["gate"] != expected_gate:
            raise ValueError("fullkv gate mismatch")
        audits[arm] = {"updates": len(steps), "question_exposures": steps[-1]["exposure_end"],
                       "feature_sha256": manifest["feature_artifact"]["sha256"],
                       "final_gate": expected_gate}
    summary = json.loads((root / "summary.json").read_text())
    if summary["protocol"] != "fullkv_head_control_v1" or summary["test_scored"]:
        raise ValueError("fullkv summary mismatch")
    if any(summary["arms"][arm]["final_gate"] != audits[arm]["final_gate"] for arm in ARMS):
        raise ValueError("fullkv summary gate mismatch")
    result = {"protocol": "fullkv_head_control_v1_audit", "passed": True,
              "data_sha256": source["data_sha256"], "arms": audits,
              "identical_head_initialization": True, "test_scored": False}
    (root / "execution_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True); parser.add_argument("--data-root", required=True)
    args = parser.parse_args(); print(json.dumps(audit(args.root, args.data_root), indent=2))


if __name__ == "__main__":
    main()
