"""One-time held-out test of the dev-selected ZsRE forgetting threshold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from tasks.evaluate_zsre_counterfactual_forgetting_controller import (
    build_scorer,
    development_gate,
    evaluate_split,
    load_split,
)
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_zsre_forgetting_controller_v2 import BinaryActionHead
from tasks.train_zsre_counterfactual_forgetting_controller_v3 import PROTOCOL as SOURCE_PROTOCOL
from tasks.diagnose_zsre_forgetting_dev_threshold import PROTOCOL as BOUNDARY_PROTOCOL


PROTOCOL = "zsre_counterfactual_forgetting_final_test_v1"


def release_threshold(boundary_summary, boundary_manifest, boundary_audit):
    if not boundary_audit.get("audit_passed"):
        raise ValueError("boundary diagnostic audit did not pass")
    if boundary_summary.get("test_loaded") or boundary_summary.get("test_scored") \
            or boundary_manifest.get("test_loaded") or boundary_manifest.get("test_scored"):
        raise ValueError("boundary diagnostic did not preserve sealed test")
    selected = boundary_summary.get("selected_passing_original_gate")
    if not selected or not selected["development_gate"]["passed"]:
        raise ValueError("boundary diagnostic has no passing development threshold")
    return float(selected["threshold"])


def run(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_native import load_model

    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    boundary, controller_root = Path(args.boundary_root), Path(args.controller_root)
    boundary_summary = json.loads((boundary / "summary.json").read_text())
    boundary_training = json.loads((boundary / "training.json").read_text())
    boundary_manifest = json.loads((boundary / "training_manifest.json").read_text())
    boundary_audit_path = Path(args.boundary_audit)
    boundary_audit = json.loads(boundary_audit_path.read_text())
    if boundary_summary["protocol"] != BOUNDARY_PROTOCOL \
            or boundary_manifest["protocol"] != BOUNDARY_PROTOCOL:
        raise ValueError("boundary protocol mismatch")
    if boundary_audit["summary_sha256"] != sha(boundary / "summary.json") \
            or boundary_audit["training_sha256"] != sha(boundary / "training.json") \
            or boundary_audit["manifest_sha256"] != sha(boundary / "training_manifest.json"):
        raise ValueError("boundary audit hashes do not match")
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

    torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    model, model_meta = load_model(args.model, "step81381_native")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    sessions = load_split(args.data_root, "test")
    scorer, feature_meta = build_scorer(
        sessions, model, tokenizer, transforms["transforms"],
        source_summary["selected_config"]["regularizer"], gate_artifact, args.batch_size)
    result, raw_hashes = evaluate_split(
        root, "test", sessions, scorer, head, artifact["mean"], artifact["std"], threshold, 9601)
    final_gate = development_gate(result["controller"], result["comparison"])
    peak_allocated = torch.cuda.max_memory_allocated() / 2**30
    peak_reserved = torch.cuda.max_memory_reserved() / 2**30
    del model; torch.cuda.empty_cache()

    summary = {
        "protocol": PROTOCOL,
        "selected_threshold": threshold,
        "test": result,
        "test_gate": final_gate,
        "test_loaded": True,
        "test_scored": True,
        "terminal_status": "completed_one_time_heldout_test",
    }
    training = {
        "seconds": time.perf_counter() - started,
        "gpu_peak_allocated_gib": peak_allocated,
        "gpu_peak_reserved_gib": peak_reserved,
        "test_features": feature_meta,
        "raw_rows_sha256": raw_hashes,
        "boundary_summary_sha256": sha(boundary / "summary.json"),
        "boundary_training_sha256": sha(boundary / "training.json"),
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
        "boundary_protocol": BOUNDARY_PROTOCOL,
        "controller_protocol": SOURCE_PROTOCOL,
        "model": args.model,
        "tokenizer": args.tokenizer,
        "loading_info": model_meta["loading_info"],
        "selected_threshold": threshold,
        "selection_rule": boundary_manifest["selection_objective"],
        "test_release_evidence": "audited passing development boundary diagnostic",
        "test_loaded": True,
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
