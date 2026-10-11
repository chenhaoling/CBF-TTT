"""Independent artifact audit for within-update counterfactual pairing."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from tasks.evaluate_cbf_candidate_ranking import answer_vocabulary
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_dynamic_memory import select_view
from tasks.train_cbf_episodic_binding import _ranking_aggregate, episodic_gate, rotation_episodes
from tasks.train_cbf_paired_binding import ARMS, PAIRINGS, paired_schedule


def audit(root: str | Path, data_root: str | Path, model: str) -> dict:
    import torch
    from transformers import AutoTokenizer

    root, data_root = Path(root), Path(data_root)
    source = json.loads((data_root / "manifest.json").read_text())
    rows_path = data_root / "rows.jsonl"
    if source["test_tokenized"] or sha(rows_path) != source["data_sha256"]:
        raise ValueError("paired source data or test seal failed")
    all_rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line]
    tokenizer = AutoTokenizer.from_pretrained(model)
    audits, initial = {}, None
    for arm in ARMS:
        smoke = json.loads((root / f"smoke_{arm}.json").read_text())
        if (smoke["arm"] != arm or smoke["test_scored"] or smoke["memory_cap"] != .3
                or smoke["peak_reserved_gib"] >= 28 or smoke["max_relative_delta"] > .305
                or not all(math.isfinite(smoke[key]) for key in
                           ("loss", "retrieval_loss", "swap_loss", "unclipped_gradient_norm"))):
            raise ValueError("invalid paired smoke")
        output = root / arm
        manifest = json.loads((output / "training_manifest.json").read_text())
        complete = json.loads((output / "complete.json").read_text())
        expected_swap = 0.0 if arm == "paired_listwise" else .5
        if (manifest["protocol"] != "paired_episodic_binding_v1" or manifest["arm"] != arm
                or manifest["data_sha256"] != source["data_sha256"] or manifest["test_scored"]
                or manifest["memory_cap"] != .3 or manifest["assignment_pairings"] != [list(x) for x in PAIRINGS]
                or manifest["context_writes_per_update"] != 2 or manifest["lm_weight"] != 0
                or manifest["swap_weight"] != expected_swap):
            raise ValueError("paired manifest mismatch")
        rows, groups = select_view(all_rows, manifest["train_groups"], manifest["dev_groups"])
        if groups != manifest["groups"]:
            raise ValueError("paired group selection mismatch")
        vocab = answer_vocabulary(rows, tokenizer)
        episodes = rotation_episodes(rows, tokenizer, vocab)
        expected = paired_schedule(rows, manifest["rounds"], manifest["order_seed"])
        steps_path = output / "steps.jsonl"
        steps = [json.loads(line) for line in steps_path.read_text().splitlines() if line]
        if (len(steps) != manifest["optimizer_updates"] or
                [(row["context_id"], *row["assignment_shifts"]) for row in steps] != expected):
            raise ValueError("paired schedule mismatch")
        coverage = {}
        for context, left, right in expected:
            coverage.setdefault(context, []).extend((left, right))
        if any(sorted(shifts) != [0, 1, 2, 3] for shifts in coverage.values()):
            raise ValueError("paired assignment coverage mismatch")
        for row, (context, left, right) in zip(steps, expected):
            if (row["left_targets"] != episodes[context][left]["targets"] or
                    row["right_targets"] != episodes[context][right]["targets"]):
                raise ValueError("paired target mismatch")
            if not 0 <= row["listwise_accuracy"] <= 1 or row["max_relative_delta"] > .305:
                raise ValueError("invalid paired metric")
            if not all(math.isfinite(row[key]) for key in
                       ("loss", "retrieval_loss", "swap_loss", "max_relative_delta")):
                raise ValueError("nonfinite paired metric")
        if (sha(steps_path) != complete["steps_sha256"] or not complete["backbone_unchanged"]
                or not complete["optimizer_state_fp32"] or complete["optimizer_updates"] != len(steps)):
            raise ValueError("paired completion mismatch")
        for key, subset, metric in (
                ("first16_retrieval_loss", steps[:16], "retrieval_loss"),
                ("last16_retrieval_loss", steps[-16:], "retrieval_loss"),
                ("first16_swap_loss", steps[:16], "swap_loss"),
                ("last16_swap_loss", steps[-16:], "swap_loss")):
            if complete[key] != statistics.fmean(row[metric] for row in subset):
                raise ValueError("paired aggregate mismatch")
        state = torch.load(output / "writer_0.pt", map_location="cpu", weights_only=True)
        if state["manifest"] != manifest:
            raise ValueError("paired checkpoint provenance mismatch")
        if initial is None:
            initial = state["writer"]
        elif set(initial) != set(state["writer"]) or any(
                not torch.equal(initial[name], state["writer"][name]) for name in initial):
            raise ValueError("paired arms do not share initialization")
        checkpoints = {}
        for exposure in manifest["checkpoints"]:
            raw = output / f"evaluation_{exposure}.jsonl"
            records = [json.loads(line) for line in raw.read_text().splitlines() if line]
            stored = json.loads((output / f"evaluation_{exposure}.summary.json").read_text())
            if len(records) != 256 or stored["rows_sha256"] != sha(raw) or stored["test_scored"]:
                raise ValueError("paired evaluation coverage mismatch")
            if _ranking_aggregate(records) != {key: stored[key] for key in ("train_probe", "dev")}:
                raise ValueError("paired evaluation aggregate mismatch")
            if stored["checkpoint_sha256"] != sha(output / f"writer_{exposure}.pt"):
                raise ValueError("paired checkpoint hash mismatch")
            checkpoints[str(exposure)] = stored
        final = checkpoints[str(manifest["question_exposures"])]
        gate = episodic_gate(final["dev"])
        if gate != final["episodic_gate"]:
            raise ValueError("paired gate mismatch")
        audits[arm] = {"smoke": smoke, "updates": len(steps),
                       "exposures": steps[-1]["exposure_end"],
                       "steps_sha256": sha(steps_path), "final_gate": gate}
    summary = json.loads((root / "summary.json").read_text())
    if summary["protocol"] != "paired_episodic_binding_v1" or summary["test_scored"]:
        raise ValueError("paired summary mismatch")
    if any(summary["arms"][arm]["final_gate"] != audits[arm]["final_gate"] for arm in ARMS):
        raise ValueError("paired summary gate mismatch")
    result = {"protocol": "paired_episodic_binding_v1_audit", "passed": True,
              "data_sha256": source["data_sha256"], "arms": audits,
              "identical_initialization": True, "test_scored": False}
    (root / "execution_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.root, args.data_root, args.model), indent=2))


if __name__ == "__main__":
    main()
