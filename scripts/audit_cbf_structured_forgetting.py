"""Independent audit for the physics-informed structured forgetting pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_structured_forgetting import (
    ACTIONS, FIXED, aggregate, build_examples, evaluate_policy, gate, load_split,
)


def audit(root: str | Path, data_root: str | Path) -> dict:
    import torch

    root, data_root = Path(root), Path(data_root)
    data_manifest = json.loads((data_root / "manifest.json").read_text())
    manifest = json.loads((root / "training_manifest.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    training = json.loads((root / "training.json").read_text())
    if (manifest["protocol"] != "structured_forgetting_v1" or manifest["update_gate"] != 1
            or tuple(manifest["actions"]) != ACTIONS or manifest["controller"] != "linear_softmax"
            or manifest["group_counts"] != {"train": 64, "dev": 16, "test": 16}
            or manifest["data_manifest_sha256"] != sha(data_root / "manifest.json")):
        raise ValueError("structured forgetting manifest mismatch")
    checkpoint = torch.load(root / "controllers.pt", map_location="cpu", weights_only=True)
    if tuple(checkpoint["actions"]) != ACTIONS or sha(root / "controllers.pt") != training["checkpoint_sha256"]:
        raise ValueError("structured forgetting checkpoint mismatch")
    models = {}
    for name in ("learned", "history_blind"):
        model = torch.nn.Linear(checkpoint["feature_dim"], len(ACTIONS))
        model.load_state_dict(checkpoint[name]); models[name] = model.eval()
    train_pairs = load_split(data_root, "train")
    full, full_meta = build_examples(train_pairs, True)
    blind, blind_meta = build_examples(train_pairs, False)
    if full_meta != training["full_examples"] or blind_meta != training["blind_examples"]:
        raise ValueError("structured forgetting label audit mismatch")
    if [row["label"] for row in full] != [row["label"] for row in blind]:
        raise ValueError("structured forgetting ablation labels differ")
    policies = FIXED + ("random", "physics", "history_blind", "learned", "oracle")
    dev_pairs = load_split(data_root, "dev")
    dev = {}
    raw_hashes = {}
    for policy in policies:
        path = root / f"dev_{policy}.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines() if line]
        recomputed = evaluate_policy(dev_pairs, policy, models)
        if records != recomputed:
            raise ValueError("structured forgetting raw rollout mismatch")
        dev[policy] = aggregate(records); raw_hashes[policy] = sha(path)
    if dev != summary["dev"] or gate(dev) != summary["dev_gate"]:
        raise ValueError("structured forgetting development summary mismatch")
    expected_test = None
    if summary["dev_gate"]["passed"]:
        test_pairs = load_split(data_root, "test")
        expected_test = {policy: aggregate(evaluate_policy(test_pairs, policy, models))
                         for policy in policies}
    if summary["test"] != expected_test or summary["test_scored"] != summary["dev_gate"]["passed"]:
        raise ValueError("structured forgetting conditional test release mismatch")
    if manifest["test_scored"] != summary["test_scored"] or summary["stage_passed"] != summary["dev_gate"]["passed"]:
        raise ValueError("structured forgetting terminal state mismatch")
    result = {"protocol": "structured_forgetting_v1_audit", "passed": True,
              "data_protocol": data_manifest["protocol"], "data_manifest_sha256": sha(data_root / "manifest.json"),
              "train_examples": len(full), "dev_raw_sha256": raw_hashes,
              "dev_gate": summary["dev_gate"], "test_scored": summary["test_scored"]}
    (root / "execution_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True); parser.add_argument("--data-root", required=True)
    args = parser.parse_args(); print(json.dumps(audit(args.root, args.data_root), indent=2))


if __name__ == "__main__":
    main()
