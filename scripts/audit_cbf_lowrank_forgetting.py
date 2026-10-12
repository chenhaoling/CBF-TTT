"""Independent recomputation audit for low-rank fast-weight forgetting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_lowrank_forgetting import (
    ACTIONS, GLOBAL_ACTIONS, POLICIES, SEEDS, Embeddings, aggregate, build_examples,
    development_gate, evaluate_policy, load_split, multi_seed_summary, restore_controllers,
)


def audit(root: str | Path, data_root: str | Path) -> dict:
    import torch

    root, data_root = Path(root), Path(data_root)
    manifest = json.loads((root / "training_manifest.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    training = json.loads((root / "training.json").read_text())
    checkpoint = torch.load(root / "controllers.pt", map_location="cpu", weights_only=True)
    if (manifest["protocol"] != "lowrank_forgetting_v1" or tuple(manifest["actions"]) != ACTIONS
            or tuple(manifest["memory_seeds"]) != SEEDS or manifest["dim"] != checkpoint["dim"]
            or manifest["global_grid"] != [0., .5, 1.]
            or manifest["data_manifest_sha256"] != sha(data_root / "manifest.json")
            or sha(root / "controllers.pt") != training["checkpoint_sha256"]):
        raise ValueError("low-rank forgetting manifest mismatch")
    train_pairs, dev_pairs = load_split(data_root, "train"), load_split(data_root, "dev")
    dev_seed, raw_hashes = {}, {}
    for seed in SEEDS:
        embeddings = Embeddings(seed, checkpoint["dim"])
        full, full_meta = build_examples(train_pairs, embeddings, ACTIONS, True)
        blind, blind_meta = build_examples(train_pairs, embeddings, ACTIONS, False)
        global_rows, global_meta = build_examples(train_pairs, embeddings, GLOBAL_ACTIONS, True)
        stored_training = training["seeds"][str(seed)]
        if (full_meta != stored_training["full"] or blind_meta != stored_training["reference_blind"]
                or global_meta != stored_training["global_only"]):
            raise ValueError("low-rank counterfactual label mismatch")
        controllers = restore_controllers(checkpoint["controllers"][str(seed)], checkpoint["dim"])
        dev_seed[str(seed)], raw_hashes[str(seed)] = {}, {}
        for policy in POLICIES:
            path = root / f"dev_seed{seed}_{policy}.jsonl"
            records = [json.loads(line) for line in path.read_text().splitlines() if line]
            recomputed = evaluate_policy(dev_pairs, policy, embeddings, controllers)
            if records != recomputed:
                raise ValueError("low-rank rollout mismatch")
            dev_seed[str(seed)][policy] = aggregate(records); raw_hashes[str(seed)][policy] = sha(path)
    dev = multi_seed_summary(dev_seed)
    if dev != summary["dev"] or development_gate(dev) != summary["dev_gate"]:
        raise ValueError("low-rank development summary mismatch")
    expected_test = None
    if summary["dev_gate"]["passed"]:
        test_pairs = load_split(data_root, "test"); test_seed = {}
        for seed in SEEDS:
            embeddings = Embeddings(seed, checkpoint["dim"])
            controllers = restore_controllers(checkpoint["controllers"][str(seed)], checkpoint["dim"])
            test_seed[str(seed)] = {policy: aggregate(evaluate_policy(test_pairs, policy, embeddings, controllers))
                                    for policy in POLICIES}
        expected_test = multi_seed_summary(test_seed)
    if summary["test"] != expected_test or summary["test_scored"] != summary["dev_gate"]["passed"]:
        raise ValueError("low-rank conditional test mismatch")
    if manifest["test_scored"] != summary["test_scored"] or summary["stage_passed"] != summary["dev_gate"]["passed"]:
        raise ValueError("low-rank terminal state mismatch")
    result = {"protocol": "lowrank_forgetting_v1_audit", "passed": True,
              "data_manifest_sha256": manifest["data_manifest_sha256"],
              "memory_seeds": list(SEEDS), "dev_raw_sha256": raw_hashes,
              "dev_gate": summary["dev_gate"], "test_scored": summary["test_scored"]}
    (root / "execution_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True); parser.add_argument("--data-root", required=True)
    args = parser.parse_args(); print(json.dumps(audit(args.root, args.data_root), indent=2))


if __name__ == "__main__":
    main()
