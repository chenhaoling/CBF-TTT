"""Zero-shot CounterFact transfer of the frozen ZsRE forgetting controller."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from tasks.build_counterfact_zero_shot_forgetting import PROTOCOL as DATA_PROTOCOL
from tasks.diagnose_zsre_forgetting_dev_threshold import PROTOCOL as BOUNDARY_PROTOCOL
from tasks.evaluate_zsre_counterfactual_forgetting_controller import build_scorer, evaluate_split
from tasks.evaluate_zsre_hard_retain_forgetting import aggregate, replay_session
from tasks.evaluate_zsre_forgetting_final_test import release_threshold
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_zsre_decomposed_scope_gate import GATE_SEEDS
from tasks.train_zsre_forgetting_controller import summarize_seed_metrics
from tasks.train_zsre_forgetting_controller_v2 import BinaryActionHead
from tasks.train_zsre_counterfactual_forgetting_controller_v3 import PROTOCOL as SOURCE_PROTOCOL


PROTOCOL = "counterfact_zero_shot_forgetting_eval_v1"
CONTROL_POLICIES = ("fixed_retain", "fixed_global", "fixed_top", "oracle_local")


def load_transfer(root):
    root = Path(root); manifest = json.loads((root / "manifest.json").read_text())
    if manifest["protocol"] != DATA_PROTOCOL or manifest["test_scored"]:
        raise ValueError("CounterFact construction protocol mismatch")
    path = root / "transfer_test.jsonl"
    if sha(path) != manifest["files_sha256"][path.name]:
        raise ValueError("CounterFact sequence hash mismatch")
    return [json.loads(line) for line in path.read_text().splitlines() if line], manifest


def transfer_gate(controller, comparison):
    checks = {
        "query_gain_confident": comparison["query_bootstrap_95ci"][0] > 0,
        "action_gain_confident": comparison["action_bootstrap_95ci"][0] > 0,
        "overall_action": controller["mean_action_accuracy"] >= .90,
        "hard_retain": controller["by_event_kind"]["hard_retain"]["accuracy"] >= .90,
        "primary_update": controller["by_event_kind"]["primary_update"]["accuracy"] >= .85,
    }
    return {"checks": checks, "passed": all(checks.values())}


def run(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_native import load_model

    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    boundary, controller_root = Path(args.boundary_root), Path(args.controller_root)
    boundary_summary = json.loads((boundary / "summary.json").read_text())
    boundary_manifest = json.loads((boundary / "training_manifest.json").read_text())
    boundary_audit_path = Path(args.boundary_audit)
    boundary_audit = json.loads(boundary_audit_path.read_text())
    if boundary_summary["protocol"] != BOUNDARY_PROTOCOL \
            or boundary_manifest["protocol"] != BOUNDARY_PROTOCOL:
        raise ValueError("boundary protocol mismatch")
    if boundary_audit["summary_sha256"] != sha(boundary / "summary.json") \
            or boundary_audit["manifest_sha256"] != sha(boundary / "training_manifest.json"):
        raise ValueError("boundary audit hash mismatch")
    threshold = release_threshold(boundary_summary, boundary_manifest, boundary_audit)

    source_manifest = json.loads((controller_root / "training_manifest.json").read_text())
    source_training = json.loads((controller_root / "training.json").read_text())
    checkpoint = controller_root / "counterfactual_forgetting_action_oof_gate.pt"
    if source_manifest["protocol"] != SOURCE_PROTOCOL \
            or source_training["checkpoint_sha256"] != sha(checkpoint):
        raise ValueError("controller source mismatch")
    artifact = torch.load(checkpoint, map_location="cpu", weights_only=False)
    head = BinaryActionHead(artifact["feature_dim"], artifact["training_seed"])
    head.load_state_dict(artifact["head"])

    source, scope = Path(args.source_root), Path(args.scope_root)
    source_summary = json.loads((source / "summary.json").read_text())
    transforms = torch.load(source / "transforms.pt", map_location="cpu", weights_only=False)
    scope_training = json.loads((scope / "training.json").read_text())
    gate_artifact = torch.load(scope / "scope_gates.pt", map_location="cpu", weights_only=False)
    if scope_training["checkpoint_sha256"] != sha(scope / "scope_gates.pt"):
        raise ValueError("scope checkpoint mismatch")
    sessions, data_manifest = load_transfer(args.data_root)

    torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    model, model_meta = load_model(args.model, "step81381_native")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    scorer, feature_meta = build_scorer(
        sessions, model, tokenizer, transforms["transforms"],
        source_summary["selected_config"]["regularizer"], gate_artifact, args.batch_size)
    primary, raw_hashes = evaluate_split(
        root, "transfer_test", sessions, scorer, head, artifact["mean"], artifact["std"],
        threshold, 9701)

    policies = {"learned": primary["controller"], "cosine_local": primary["cosine_local"]}
    for policy in CONTROL_POLICIES:
        per_seed = {}
        for seed in GATE_SEEDS:
            records = [replay_session(session, policy, scorer, seed) for session in sessions]
            for record in records:
                record["split"] = "transfer_test"
            per_seed[str(seed)] = aggregate(records)
            path = root / f"transfer_test_{policy}_seed{seed}.jsonl"
            path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
            raw_hashes[path.name] = sha(path)
        policies[policy] = summarize_seed_metrics(per_seed)
    gate = transfer_gate(primary["controller"], primary["comparison"])
    peak_allocated = torch.cuda.max_memory_allocated() / 2**30
    peak_reserved = torch.cuda.max_memory_reserved() / 2**30
    del model; torch.cuda.empty_cache()

    summary = {
        "protocol": PROTOCOL,
        "selected_threshold": threshold,
        "units": data_manifest["eligible_subjects"],
        "policies": policies,
        "primary_comparison": primary["comparison"],
        "transfer_gate": gate,
        "counterfact_training_rows": 0,
        "counterfact_tuning_rows": 0,
        "test_scored": True,
        "terminal_status": "completed_counterfact_zero_shot_transfer",
    }
    training = {
        "seconds": time.perf_counter() - started,
        "gpu_peak_allocated_gib": peak_allocated,
        "gpu_peak_reserved_gib": peak_reserved,
        "features": feature_meta,
        "raw_rows_sha256": raw_hashes,
        "data_manifest_sha256": sha(Path(args.data_root) / "manifest.json"),
        "boundary_summary_sha256": sha(boundary / "summary.json"),
        "boundary_manifest_sha256": sha(boundary / "training_manifest.json"),
        "boundary_audit_sha256": sha(boundary_audit_path),
        "controller_checkpoint_sha256": sha(checkpoint),
        "source_summary_sha256": sha(source / "summary.json"),
        "source_transforms_sha256": sha(source / "transforms.pt"),
        "scope_training_sha256": sha(scope / "training.json"),
        "scope_checkpoint_sha256": sha(scope / "scope_gates.pt"),
    }
    manifest = {
        "protocol": PROTOCOL,
        "data_protocol": DATA_PROTOCOL,
        "controller_protocol": SOURCE_PROTOCOL,
        "boundary_protocol": BOUNDARY_PROTOCOL,
        "model": args.model,
        "tokenizer": args.tokenizer,
        "loading_info": model_meta["loading_info"],
        "selected_threshold": threshold,
        "control_policies": list(CONTROL_POLICIES),
        "primary_control": "cosine_local",
        "selection_split": "ZsRE development only",
        "counterfact_training_rows": 0,
        "counterfact_tuning_rows": 0,
        "test_scored": True,
    }
    write_json(root / "summary.json", summary)
    write_json(root / "training.json", training)
    write_json(root / "training_manifest.json", manifest)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--boundary-root", required=True)
    parser.add_argument("--boundary-audit", required=True)
    parser.add_argument("--controller-root", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--scope-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
