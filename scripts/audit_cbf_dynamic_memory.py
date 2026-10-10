"""Independent artifact/accounting audit for the dynamic memory writer pilot."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_dynamic_memory import ARMS, aggregate, schedule, select_view, summarize
from tasks.train_cbf_event_writer import write_json


def contains(sequence, subsequence):
    size = len(subsequence)
    return any(sequence[index:index + size] == subsequence for index in range(len(sequence) - size + 1))


def audit_data(data_root: Path) -> tuple[list[dict], dict, dict]:
    manifest = json.loads((data_root / "manifest.json").read_text())
    source_root = Path(manifest["source"])
    source_manifest = json.loads((source_root / "manifest.json").read_text())
    if manifest["protocol"] != "dynamic_memory_writer_v1" or manifest["test_tokenized"]:
        raise ValueError("data protocol or test seal mismatch")
    if sha(data_root / "rows.jsonl") != manifest["data_sha256"]:
        raise ValueError("packed data hash mismatch")
    if sha(data_root / "backgrounds.json") != manifest["backgrounds_sha256"]:
        raise ValueError("background manifest hash mismatch")
    if sha(source_root / "manifest.json") != manifest["source_manifest_sha256"]:
        raise ValueError("semantic source manifest mismatch")
    rows = [json.loads(line) for line in (data_root / "rows.jsonl").read_text().splitlines() if line]
    if {row["split"] for row in rows} != {"train", "dev"}:
        raise ValueError("sealed test appeared in packed rows")
    expected = sum(manifest["rows"].values())
    if len(rows) != expected or any(len(row["context_ids"]) != manifest["chunk"] for row in rows):
        raise ValueError("packed matrix shape mismatch")
    by_id = {row["id"]: row for row in rows}
    contexts = defaultdict(list)
    backgrounds = defaultdict(set)
    for row in rows:
        contexts[row["context_id"]].append(row)
        backgrounds[(row["split"], row["group_id"])].add(row["background_sha256"])
        if not contains(row["context_ids"], row["answer_ids"]):
            raise ValueError("written answer absent from context tokens")
        if row["wrong_context_id"].split(".")[0] == row["group_id"]:
            raise ValueError("wrong memory is not cross-world")
        twin = by_id[row["twin_row_id"]]
        if twin["query_ids"] != row["query_ids"]:
            raise ValueError("twin query differs")
        if (twin["answer"] != row["answer"]) != row["anchor_query"]:
            raise ValueError("twin causal change mismatch")
    if any(len(items) != 4 for items in contexts.values()) or any(len(values) != 1 for values in backgrounds.values()):
        raise ValueError("context supervision or background reuse mismatch")
    background_rows = json.loads((data_root / "backgrounds.json").read_text())
    if len({item["text_sha256"] for item in background_rows}) != len(background_rows):
        raise ValueError("background documents are not unique")
    if Counter(item["source"] for item in background_rows) != Counter(manifest["background_sources"]):
        raise ValueError("background source count mismatch")
    if source_manifest["group_counts"] != {**manifest["groups"], "test": source_manifest["group_counts"]["test"]}:
        raise ValueError("semantic and packed group counts differ")
    return rows, manifest, source_manifest


def load_checkpoint(path: Path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=True)


def audit_arm(root: Path, rows: list[dict], arm: str) -> dict:
    output = root / arm
    manifest = json.loads((output / "training_manifest.json").read_text())
    complete = json.loads((output / "complete.json").read_text())
    selected_rows, selected_groups = select_view(rows,
        len(manifest["groups"]["train"]), len(manifest["groups"]["dev"]))
    if selected_groups != manifest["groups"] or manifest["arm"] != arm or manifest["test_scored"]:
        raise ValueError("training manifest mismatch")
    expected = schedule(selected_rows, arm, manifest["rounds"], manifest["order_seed"])
    steps = [json.loads(line) for line in (output / "steps.jsonl").read_text().splitlines() if line]
    if [row["ids"] for row in steps] != expected:
        raise ValueError("optimizer schedule mismatch")
    if complete["steps_sha256"] != sha(output / "steps.jsonl"):
        raise ValueError("step hash mismatch")
    if complete["optimizer_updates"] != len(expected) or complete["question_exposures"] != sum(map(len, expected)):
        raise ValueError("training budget mismatch")
    if not all(math.isfinite(row["loss"]) and row["exposure_end"] - row["exposure_start"] == len(row["ids"])
               for row in steps):
        raise ValueError("nonfinite loss or exposure accounting")
    initial = load_checkpoint(output / "writer_0.pt")
    final = load_checkpoint(output / f"writer_{manifest['question_exposures']}.pt")
    if initial["manifest"] != manifest or final["manifest"] != manifest:
        raise ValueError("checkpoint provenance mismatch")
    import torch
    for name in initial["writer"]:
        delta = float((final["writer"][name] - initial["writer"][name]).norm())
        if delta <= 0 or not math.isclose(delta, complete["writer_update_norms"][name], rel_tol=2e-6, abs_tol=1e-8):
            raise ValueError("saved writer update mismatch")
        if initial["writer"][name].dtype != torch.float32 or final["writer"][name].dtype != torch.float32:
            raise ValueError("writer checkpoint is not FP32")
    optimizer = load_checkpoint(output / f"optimizer_{manifest['question_exposures']}.pt")
    if not all(value.dtype == torch.float32 for state in optimizer["state"].values()
               for value in state.values() if isinstance(value, torch.Tensor)):
        raise ValueError("optimizer checkpoint is not FP32")
    evaluation_rows = 8 * 8 + len(manifest["groups"]["dev"]) * 8
    resources = []
    for exposure in manifest["checkpoints"]:
        raw_path = output / f"evaluation_{exposure}.jsonl"
        raw = [json.loads(line) for line in raw_path.read_text().splitlines() if line]
        stored = json.loads((output / f"evaluation_{exposure}.summary.json").read_text())
        if len(raw) != evaluation_rows or stored["rows_sha256"] != sha(raw_path):
            raise ValueError("evaluation coverage/hash mismatch")
        if aggregate(raw) != {key: stored[key] for key in ("train_probe", "dev")}:
            raise ValueError("evaluation aggregate mismatch")
        for row in raw:
            required = {"correct", "empty", "wrong", "full_kv"}
            if set(row["policies"]) != required | ({"twin"} if row["anchor_query"] else set()):
                raise ValueError("evaluation policy matrix mismatch")
            resources.extend(row["policies"].values())
    return {
        "arm": arm,
        "updates": len(steps),
        "question_exposures": complete["question_exposures"],
        "peak_training_allocated_gib": complete["peak_allocated_gib"],
        "peak_training_reserved_gib": complete["peak_reserved_gib"],
        "max_evaluation_seconds": max(item["seconds"] for item in resources),
        "max_evaluation_allocated_gib": max(item["peak_allocated_gib"] for item in resources),
        "backbone_sha256": manifest["backbone_sha256"],
        "initial": initial,
    }


def audit(root: Path, data_root: Path) -> dict:
    rows, data_manifest, source_manifest = audit_data(data_root)
    arm_reports = {arm: audit_arm(root, rows, arm) for arm in ARMS}
    left, right = (arm_reports[arm]["initial"] for arm in ARMS)
    if set(left["writer"]) != set(right["writer"]) or any(
        not left["writer"][name].equal(right["writer"][name]) for name in left["writer"]
    ):
        raise ValueError("arms did not start from identical writer tensors")
    if arm_reports["sequential"]["backbone_sha256"] != arm_reports["joint_context"]["backbone_sha256"]:
        raise ValueError("arms used different frozen backbones")
    for report in arm_reports.values():
        report.pop("initial")
    stored = json.loads((root / "summary.json").read_text())
    if summarize(root, save=False) != stored:
        raise ValueError("top-level summary mismatch")
    result = {
        "protocol": "dynamic_memory_writer_v1_audit",
        "passed": True,
        "data": {
            "rows": len(rows), "data_sha256": data_manifest["data_sha256"],
            "backgrounds_sha256": data_manifest["backgrounds_sha256"],
            "semantic_seed": source_manifest["seed"], "test_tokenized": False,
        },
        "arms": arm_reports,
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

