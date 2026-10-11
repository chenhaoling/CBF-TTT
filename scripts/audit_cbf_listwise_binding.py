"""Independent no-model audit for candidate-aligned listwise binding."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_associative_binding import aggregate, schedule
from tasks.train_cbf_listwise_binding import ARMS, context_candidates, optimization_gate


def audit(root: str | Path, data_root: str | Path) -> dict:
    import torch

    root, data_root = Path(root), Path(data_root)
    data_manifest = json.loads((data_root / "manifest.json").read_text())
    rows_path = data_root / "rows.jsonl"
    if data_manifest["test_tokenized"] or sha(rows_path) != data_manifest["data_sha256"]:
        raise ValueError("listwise source data or test sealing failed")
    all_rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line]
    audits = {}
    initial = None
    for arm in ARMS:
        smoke = json.loads((root / f"smoke_{arm}.json").read_text())
        if (smoke["arm"] != arm or smoke["test_scored"] or smoke["max_relative_delta"] > 1
                or not all(math.isfinite(smoke[key]) for key in
                           ("loss", "retrieval_loss", "balanced_loss", "unclipped_gradient_norm"))):
            raise ValueError("invalid listwise smoke")
        output = root / arm
        manifest = json.loads((output / "training_manifest.json").read_text())
        complete = json.loads((output / "complete.json").read_text())
        if (manifest["protocol"] != "norm_listwise_binding_v1" or manifest["arm"] != arm
                or manifest["data_sha256"] != data_manifest["data_sha256"]
                or manifest["test_scored"] or manifest["context_writes_per_update"] != 1
                or manifest["listwise_candidates"] != 4 or manifest["memory_cap"] != .3):
            raise ValueError("listwise manifest mismatch")
        selected_groups = manifest["groups"]
        rows = [row for row in all_rows if row["group_id"] in selected_groups[row["split"]]]
        candidates = context_candidates(rows)
        expected = schedule(rows, manifest["rounds"], manifest["order_seed"])
        by_id = {row["id"]: row for row in rows}
        steps_path = output / "steps.jsonl"
        steps = [json.loads(line) for line in steps_path.read_text().splitlines() if line]
        if len(steps) != manifest["optimizer_updates"] or [row["ids"] for row in steps] != expected:
            raise ValueError("listwise training schedule mismatch")
        for row in steps:
            truths = {by_id[sample_id]["answer"] for sample_id in row["ids"]}
            if set(row["candidate_values"]) != truths or row["candidate_values"] != candidates[row["context_id"]]:
                raise ValueError("listwise candidate set mismatch")
            if not 0 <= row["listwise_accuracy"] <= 1:
                raise ValueError("invalid listwise training accuracy")
        if any(not math.isfinite(row[key]) for row in steps for key in
               ("loss", "retrieval_loss", "balanced_loss", "max_relative_delta")):
            raise ValueError("nonfinite listwise training metric")
        if max(row["max_relative_delta"] for row in steps) > 1:
            raise ValueError("listwise safety tripwire exceeded")
        if (sha(steps_path) != complete["steps_sha256"] or not complete["backbone_unchanged"]
                or complete["optimizer_updates"] != len(steps)):
            raise ValueError("listwise completion mismatch")
        for key, values in (
                ("first32_retrieval_loss", steps[:32]), ("last32_retrieval_loss", steps[-32:])):
            if complete[key] != statistics.fmean(row["retrieval_loss"] for row in values):
                raise ValueError("listwise loss aggregate mismatch")
        state = torch.load(output / "writer_0.pt", map_location="cpu", weights_only=True)
        if state["manifest"] != manifest:
            raise ValueError("listwise initial checkpoint provenance mismatch")
        if initial is None:
            initial = state["writer"]
        elif set(initial) != set(state["writer"]) or any(
                not torch.equal(initial[name], state["writer"][name]) for name in initial):
            raise ValueError("listwise arms do not share initialization")
        checkpoints = {}
        for exposure in manifest["checkpoints"]:
            raw = output / f"evaluation_{exposure}.jsonl"
            records = [json.loads(line) for line in raw.read_text().splitlines() if line]
            stored = json.loads((output / f"evaluation_{exposure}.summary.json").read_text())
            if len(records) != 128 or stored["rows_sha256"] != sha(raw):
                raise ValueError("listwise evaluation coverage mismatch")
            if aggregate(records) != {key: stored[key] for key in ("train_probe", "dev")}:
                raise ValueError("listwise evaluation aggregate mismatch")
            if stored["checkpoint_sha256"] != sha(output / f"writer_{exposure}.pt") or stored["test_scored"]:
                raise ValueError("listwise checkpoint or test seal mismatch")
            checkpoints[str(exposure)] = stored
        final = checkpoints[str(manifest["question_exposures"])]
        gate = optimization_gate(complete, final["dev"], final["train_probe"])
        audits[arm] = {
            "smoke": smoke, "updates": len(steps), "exposures": steps[-1]["exposure_end"],
            "steps_sha256": sha(steps_path), "final_gate": gate,
        }
    summary = json.loads((root / "summary.json").read_text())
    if summary["protocol"] != "norm_listwise_binding_v1" or summary["test_scored"]:
        raise ValueError("listwise summary mismatch")
    if any(summary["arms"][arm]["final_gate"] != audits[arm]["final_gate"] for arm in ARMS):
        raise ValueError("listwise gate recomputation mismatch")
    if summary["stage_passed"] != audits["retrieval_plus_lm"]["final_gate"]["passed"]:
        raise ValueError("listwise stage decision mismatch")
    result = {
        "protocol": "norm_listwise_binding_v1_audit", "passed": True,
        "data_sha256": data_manifest["data_sha256"], "arms": audits,
        "identical_initialization": True, "test_scored": False,
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
