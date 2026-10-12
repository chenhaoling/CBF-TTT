"""Train forgetting control on the frozen, gated contextual Qwen memory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_cbf_lowrank_forgetting import (
    ACTIONS, GLOBAL_ACTIONS, POLICIES, aggregate, build_examples, evaluate_policy,
    restore_controllers, serialize_controller, train_controller,
)
from tasks.train_cbf_qwen_lowrank_forgetting import SEEDS, development_gate, multi_seed_summary
from tasks.train_cbf_structured_forgetting import load_split
from tasks.diagnose_cbf_qwen_contextual_geometry import (
    apply_contextual_transform, extract_contextual,
)


def run(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_native import load_model

    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    source = Path(args.contextual_root)
    source_summary = json.loads((source / "summary.json").read_text())
    source_training = json.loads((source / "training.json").read_text())
    source_manifest = json.loads((source / "training_manifest.json").read_text())
    if (source_summary["protocol"] != "qwen_contextual_geometry_v1"
            or not source_summary["stage_passed"]
            or source_summary["selected_variant"] != "context_last"
            or source_summary["test_scored"]):
        raise ValueError("contextual source did not pass the registered development gate")
    config = source_summary["selected_configs"]["context_last"]
    if config["internal"] != "context_last_r0.01" or config["unknown_raw"] != .75:
        raise ValueError("unexpected frozen contextual configuration")
    if source_training["transforms_sha256"] != sha(source / "contextual_transforms.pt"):
        raise ValueError("contextual transform hash mismatch")
    transform_artifact = torch.load(source / "contextual_transforms.pt", map_location="cpu",
                                    weights_only=False)
    transforms = transform_artifact["transforms"]

    train_pairs = load_split(args.data_root, "train")
    dev_pairs = load_split(args.data_root, "dev")
    torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    model, model_meta = load_model(args.model, "step81381_native")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    raw, names, extraction_meta = extract_contextual(
        model, tokenizer, train_pairs + dev_pairs, args.batch_size)
    embeddings, embedding_hashes = apply_contextual_transform(
        raw, names, transforms, "context_last", config["regularizer"])
    expected_hashes = source_training["context_embedding_meta"]["hashes"][config["internal"]]
    if embedding_hashes != expected_hashes:
        raise ValueError("contextual train/dev embedding bridge mismatch")
    del model, raw; torch.cuda.empty_cache()

    controllers_saved, training, dev_seed = {}, {}, {}
    for seed in SEEDS:
        memory = embeddings[str(seed)]
        full, full_meta = build_examples(
            train_pairs, memory, ACTIONS, True, config["unknown_raw"])
        blind, blind_meta = build_examples(
            train_pairs, memory, ACTIONS, False, config["unknown_raw"])
        global_rows, global_meta = build_examples(
            train_pairs, memory, GLOBAL_ACTIONS, True, config["unknown_raw"])
        learned, curve = train_controller(full, ACTIONS, args.seed + seed, args.epochs)
        blind_model, blind_curve = train_controller(blind, ACTIONS, args.seed + seed, args.epochs)
        global_model, global_curve = train_controller(
            global_rows, GLOBAL_ACTIONS, args.seed + seed, args.epochs)
        controllers = {"learned": learned, "reference_blind": blind_model,
                       "global_learned": global_model}
        controllers_saved[str(seed)] = {
            name: serialize_controller(value) for name, value in controllers.items()}
        training[str(seed)] = {
            "full": full_meta, "reference_blind": blind_meta, "global_only": global_meta,
            "curve": curve, "reference_blind_curve": blind_curve, "global_curve": global_curve,
        }
        dev_seed[str(seed)] = {}
        for policy in POLICIES:
            records = evaluate_policy(
                dev_pairs, policy, memory, controllers, config["unknown_raw"])
            dev_seed[str(seed)][policy] = aggregate(records)
            path = root / f"dev_seed{seed}_{policy}.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in records))
    dev = multi_seed_summary(dev_seed); gate = development_gate(dev)

    test_scored, test, test_embedding_meta = False, None, None
    if gate["passed"]:
        test_pairs = load_split(args.data_root, "test")
        model, _ = load_model(args.model, "step81381_native")
        test_raw, test_names, test_extraction = extract_contextual(
            model, tokenizer, test_pairs, args.batch_size)
        test_embeddings, test_hashes = apply_contextual_transform(
            test_raw, test_names, transforms, "context_last", config["regularizer"])
        test_embedding_meta = {"hashes": test_hashes, "extraction": test_extraction}
        del model, test_raw; torch.cuda.empty_cache()
        test_seed = {}
        for seed in SEEDS:
            memory = test_embeddings[str(seed)]
            controllers = restore_controllers(controllers_saved[str(seed)], memory.dim)
            test_seed[str(seed)] = {}
            for policy in POLICIES:
                records = evaluate_policy(
                    test_pairs, policy, memory, controllers, config["unknown_raw"])
                test_seed[str(seed)][policy] = aggregate(records)
                path = root / f"test_seed{seed}_{policy}.jsonl"
                path.write_text("".join(json.dumps(row) + "\n" for row in records))
        test = multi_seed_summary(test_seed); test_scored = True

    checkpoint = root / "controllers.pt"
    torch.save({"controllers": controllers_saved, "dim": 256, "seeds": SEEDS,
                "epochs": args.epochs, "seed": args.seed, "contextual_config": config}, checkpoint)
    raw_hashes = {path.name: sha(path) for path in root.glob("*.jsonl")}
    training_result = {
        "seconds": time.perf_counter() - started,
        "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "seeds": training, "checkpoint_sha256": sha(checkpoint),
        "contextual_config": config, "embedding_hashes": embedding_hashes,
        "test_embedding_meta": test_embedding_meta, "raw_rows_sha256": raw_hashes,
        "source_summary_sha256": sha(source / "summary.json"),
        "source_training_sha256": sha(source / "training.json"),
        "source_manifest_sha256": sha(source / "training_manifest.json"),
        "source_transforms_sha256": sha(source / "contextual_transforms.pt"),
        "train_dev_embedding_bridge_exact": True, "extraction_meta": extraction_meta,
    }
    manifest = {
        "protocol": "qwen_contextual_forgetting_v2",
        "data_manifest_sha256": sha(Path(args.data_root) / "manifest.json"),
        "model": args.model, "tokenizer": args.tokenizer,
        "loading_info": model_meta["loading_info"], "memory_seeds": list(SEEDS),
        "dim": 256, "actions": list(ACTIONS), "global_grid": [0., .5, 1.],
        "contextual_config": config, "source_protocol": source_manifest["protocol"],
        "test_release_rule": "score once only after unchanged aggregate development gate",
        "test_scored": test_scored,
    }
    result = {
        "protocol": "qwen_contextual_forgetting_v2", "dev": dev,
        "dev_gate": gate, "test": test, "test_scored": test_scored,
        "stage_passed": gate["passed"], "training_seconds": training_result["seconds"],
        "terminal_status": "completed_qwen_contextual_forgetting_v2",
    }
    write_json(root / "training.json", training_result)
    write_json(root / "training_manifest.json", manifest)
    write_json(root / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--contextual-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2321)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
