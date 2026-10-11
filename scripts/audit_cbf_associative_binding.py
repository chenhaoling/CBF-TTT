"""Independent no-model audit for the all-fact associative binding pilot."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_associative_binding import ARMS, aggregate, schedule


def audit(root: str | Path, data_root: str | Path) -> dict:
    import torch

    root, data_root = Path(root), Path(data_root)
    data_manifest = json.loads((data_root / "manifest.json").read_text())
    if data_manifest["test_tokenized"] or sha(data_root / "rows.jsonl") != data_manifest["data_sha256"]:
        raise ValueError("source data integrity or test sealing failed")
    all_rows = [json.loads(line) for line in (data_root / "rows.jsonl").read_text().splitlines() if line]
    audits = {}
    initial = None
    for arm in ARMS:
        output = root / arm
        manifest = json.loads((output / "training_manifest.json").read_text())
        complete = json.loads((output / "complete.json").read_text())
        if manifest["protocol"] != "associative_binding_v1" or manifest["arm"] != arm:
            raise ValueError("manifest arm mismatch")
        if manifest["data_sha256"] != data_manifest["data_sha256"] or manifest["test_scored"]:
            raise ValueError("manifest provenance mismatch")
        if manifest["train_groups"] != 32 or manifest["dev_groups"] != 8:
            raise ValueError("pilot group count changed")
        selected_groups = manifest["groups"]
        rows = [row for row in all_rows if row["group_id"] in selected_groups[row["split"]]]
        expected = schedule(rows, manifest["rounds"], manifest["order_seed"])
        steps_path = output / "steps.jsonl"
        steps = [json.loads(line) for line in steps_path.read_text().splitlines() if line]
        if len(steps) != manifest["optimizer_updates"] or [row["ids"] for row in steps] != expected:
            raise ValueError("training schedule mismatch")
        expected_terms = 1 if arm == "anchor_twin" else 4
        if manifest["contrast_questions"] != expected_terms or any(
                len(row["contrast_terms"]) != expected_terms for row in steps):
            raise ValueError("contrast question count mismatch")
        by_id = {row["id"]: row for row in rows}
        for row in steps:
            original = [by_id[sample_id]["answer"] for sample_id in row["ids"]]
            if (len(set(row["permuted_answers"])) != 4
                    or set(row["permuted_answers"]) != set(original)
                    or any(left == right for left, right in zip(original, row["permuted_answers"]))):
                raise ValueError("permutation is not a four-value derangement")
        numeric = [row[key] for row in steps for key in
                   ("loss", "balanced_loss", "contrast_loss", "max_relative_delta")]
        if not all(math.isfinite(value) for value in numeric) or max(
                row["max_relative_delta"] for row in steps) > 1:
            raise ValueError("nonfinite or unsafe training record")
        if sha(steps_path) != complete["steps_sha256"] or not complete["backbone_unchanged"]:
            raise ValueError("completion integrity mismatch")
        state = torch.load(output / "writer_0.pt", map_location="cpu", weights_only=True)
        if state["manifest"] != manifest:
            raise ValueError("initial checkpoint manifest mismatch")
        if initial is None:
            initial = state["writer"]
        elif set(initial) != set(state["writer"]) or any(
                not torch.equal(initial[name], state["writer"][name]) for name in initial):
            raise ValueError("arms do not share identical initialization")
        checkpoints = {}
        for exposure in manifest["checkpoints"]:
            raw = output / f"evaluation_{exposure}.jsonl"
            records = [json.loads(line) for line in raw.read_text().splitlines() if line]
            stored = json.loads((output / f"evaluation_{exposure}.summary.json").read_text())
            if len(records) != 128 or stored["rows_sha256"] != sha(raw):
                raise ValueError("evaluation coverage mismatch")
            if aggregate(records) != {key: stored[key] for key in ("train_probe", "dev")}:
                raise ValueError("evaluation aggregation mismatch")
            if stored["checkpoint_sha256"] != sha(output / f"writer_{exposure}.pt"):
                raise ValueError("evaluation checkpoint hash mismatch")
            if stored["test_scored"]:
                raise ValueError("test was scored")
            checkpoints[str(exposure)] = stored["ranking_gate"]
        audits[arm] = {
            "updates": len(steps), "exposures": steps[-1]["exposure_end"],
            "contrast_questions": expected_terms, "checkpoints": checkpoints,
            "steps_sha256": sha(steps_path),
        }
    summary = json.loads((root / "summary.json").read_text())
    if summary["protocol"] != "associative_binding_v1" or summary["test_scored"]:
        raise ValueError("summary protocol mismatch")
    primary = summary["arms"]["all_fact_permutation"]["final_gate"]
    if summary["stage_passed"] != primary["passed"]:
        raise ValueError("stage decision mismatch")
    result = {
        "protocol": "associative_binding_v1_audit", "passed": True,
        "data_sha256": data_manifest["data_sha256"], "arms": audits,
        "identical_initialization": True, "primary_gate": primary,
        "test_scored": False,
    }
    (root / "execution_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.root, args.data_root), indent=2))


if __name__ == "__main__":
    main()
