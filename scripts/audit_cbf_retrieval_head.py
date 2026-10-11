"""Independent artifact audit for the episodic candidate retrieval head."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_retrieval_head import ARMS, _aggregate, gate, head_schedule


def audit(root: str | Path, data_root: str | Path, model: str) -> dict:
    import torch
    from transformers import AutoTokenizer
    from tasks.evaluate_cbf_candidate_ranking import answer_vocabulary
    from tasks.train_cbf_dynamic_memory import select_view
    from tasks.train_cbf_episodic_binding import rotation_episodes

    root, data_root = Path(root), Path(data_root)
    source = json.loads((data_root / "manifest.json").read_text())
    rows_path = data_root / "rows.jsonl"
    if source["test_tokenized"] or sha(rows_path) != source["data_sha256"]:
        raise ValueError("retrieval-head source or test seal failed")
    all_rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line]
    tokenizer = AutoTokenizer.from_pretrained(model)
    initial_writer = initial_head = None
    audits = {}
    for arm in ARMS:
        smoke = json.loads((root / f"smoke_{arm}.json").read_text())
        if (smoke["arm"] != arm or smoke["test_scored"] or smoke["peak_reserved_gib"] >= 28
                or smoke["max_relative_delta"] > .305 or
                not all(math.isfinite(smoke[key]) for key in
                        ("loss", "accuracy", "head_gradient_norm", "writer_gradient_norm"))):
            raise ValueError("invalid retrieval-head smoke")
        output = root / arm
        manifest = json.loads((output / "training_manifest.json").read_text())
        complete = json.loads((output / "complete.json").read_text())
        if (manifest["protocol"] != "retrieval_head_v1" or manifest["arm"] != arm
                or manifest["data_sha256"] != source["data_sha256"] or manifest["test_scored"]
                or manifest["projection_size"] != 256 or manifest["temperature"] != .1
                or manifest["head_lr"] != 1e-4 or manifest["writer_lr"] != 1e-7
                or manifest["memory_cap"] != .3
                or manifest["writer_trainable"] != (arm == "joint_head_writer")):
            raise ValueError("retrieval-head manifest mismatch")
        rows, groups = select_view(all_rows, manifest["train_groups"], manifest["dev_groups"])
        if groups != manifest["groups"]:
            raise ValueError("retrieval-head groups mismatch")
        vocab = answer_vocabulary(rows, tokenizer)
        rotation_episodes(rows, tokenizer, vocab)
        expected = head_schedule(rows, manifest["cycles"], manifest["order_seed"])
        steps_path = output / "steps.jsonl"
        steps = [json.loads(line) for line in steps_path.read_text().splitlines() if line]
        if (len(steps) != manifest["optimizer_updates"] or
                [(row["context_id"], *row["assignment_shifts"]) for row in steps] != expected):
            raise ValueError("retrieval-head schedule mismatch")
        if any(not 0 <= row["accuracy"] <= 1 or row["max_relative_delta"] > .305 for row in steps):
            raise ValueError("invalid retrieval-head metric")
        if any(not all(math.isfinite(row[key]) for key in
                       ("loss", "accuracy", "head_gradient_norm", "writer_gradient_norm",
                        "max_relative_delta")) for row in steps):
            raise ValueError("nonfinite retrieval-head metric")
        if arm == "head_only" and any(row["writer_gradient_norm"] != 0 for row in steps):
            raise ValueError("head-only writer received gradient")
        if arm == "joint_head_writer" and any(row["writer_gradient_norm"] <= 0 for row in steps):
            raise ValueError("joint writer missing gradient")
        if (sha(steps_path) != complete["steps_sha256"] or not complete["backbone_unchanged"]
                or not complete["optimizer_state_fp32"] or complete["optimizer_updates"] != len(steps)):
            raise ValueError("retrieval-head completion mismatch")
        for key, subset, metric in (("first32_loss", steps[:32], "loss"),
                                    ("last32_loss", steps[-32:], "loss"),
                                    ("first32_accuracy", steps[:32], "accuracy"),
                                    ("last32_accuracy", steps[-32:], "accuracy")):
            if complete[key] != statistics.fmean(row[metric] for row in subset):
                raise ValueError("retrieval-head aggregate mismatch")
        zero = torch.load(output / "checkpoint_0.pt", map_location="cpu", weights_only=True)
        final = torch.load(output / f"checkpoint_{manifest['question_exposures']}.pt",
                           map_location="cpu", weights_only=True)
        if zero["manifest"] != manifest or final["manifest"] != manifest:
            raise ValueError("retrieval-head checkpoint provenance mismatch")
        if initial_writer is None:
            initial_writer, initial_head = zero["writer"], zero["head"]
        elif (any(not torch.equal(initial_writer[name], zero["writer"][name])
                  for name in initial_writer) or
              any(not torch.equal(initial_head[name], zero["head"][name]) for name in initial_head)):
            raise ValueError("retrieval-head arms do not share initialization")
        writer_changed = {name: not torch.equal(zero["writer"][name], final["writer"][name])
                          for name in zero["writer"]}
        head_changed = {name: not torch.equal(zero["head"][name], final["head"][name])
                        for name in zero["head"]}
        if not all(head_changed.values()) or (arm == "head_only" and any(writer_changed.values())):
            raise ValueError("retrieval-head parameter change mismatch")
        if arm == "joint_head_writer" and not all(writer_changed.values()):
            raise ValueError("joint writer did not update")
        checkpoint_summaries = {}
        for exposure in manifest["checkpoints"]:
            raw = output / f"evaluation_{exposure}.jsonl"
            records = [json.loads(line) for line in raw.read_text().splitlines() if line]
            stored = json.loads((output / f"evaluation_{exposure}.summary.json").read_text())
            if len(records) != 256 or stored["rows_sha256"] != sha(raw) or stored["test_scored"]:
                raise ValueError("retrieval-head evaluation coverage mismatch")
            if _aggregate(records) != {key: stored[key] for key in ("train_probe", "dev")}:
                raise ValueError("retrieval-head evaluation aggregate mismatch")
            checkpoint_summaries[str(exposure)] = stored
        final_summary = checkpoint_summaries[str(manifest["question_exposures"])]
        final_gate = gate(final_summary["dev"])
        if final_gate != final_summary["gate"]:
            raise ValueError("retrieval-head gate mismatch")
        audits[arm] = {"smoke": smoke, "updates": len(steps),
                       "exposures": steps[-1]["exposure_end"],
                       "steps_sha256": sha(steps_path), "final_gate": final_gate}
    summary = json.loads((root / "summary.json").read_text())
    if summary["protocol"] != "retrieval_head_v1" or summary["test_scored"]:
        raise ValueError("retrieval-head summary mismatch")
    if any(summary["arms"][arm]["final_gate"] != audits[arm]["final_gate"] for arm in ARMS):
        raise ValueError("retrieval-head summary gate mismatch")
    result = {"protocol": "retrieval_head_v1_audit", "passed": True,
              "data_sha256": source["data_sha256"], "arms": audits,
              "identical_writer_and_head_initialization": True, "test_scored": False}
    (root / "execution_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True); parser.add_argument("--data-root", required=True)
    parser.add_argument("--model", required=True); args = parser.parse_args()
    print(json.dumps(audit(args.root, args.data_root, args.model), indent=2))


if __name__ == "__main__": main()
