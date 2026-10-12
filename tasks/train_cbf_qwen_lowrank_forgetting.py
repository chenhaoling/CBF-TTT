"""Physics-informed forgetting on Qwen-grounded projected low-rank memory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_cbf_lowrank_forgetting import (
    ACTIONS, GLOBAL_ACTIONS, POLICIES, apply_action, batch_delta, build_examples,
    evaluate_policy, aggregate,
    restore_controllers, serialize_controller, train_controller,
)
from tasks.train_cbf_associative_binding import _bootstrap
from tasks.train_cbf_structured_forgetting import load_split


SEEDS = (2301, 2302, 2303)


def multi_seed_summary(per_seed):
    policies = next(iter(per_seed.values())).keys(); output = {}
    for policy in policies:
        output[policy] = {"mean_accuracy": statistics.fmean(per_seed[str(seed)][policy]["accuracy"] for seed in SEEDS),
                          "mean_nll": statistics.fmean(per_seed[str(seed)][policy]["nll"] for seed in SEEDS),
                          "min_seed_accuracy": min(per_seed[str(seed)][policy]["accuracy"] for seed in SEEDS),
                          "per_seed": {str(seed): per_seed[str(seed)][policy] for seed in SEEDS}}
    return output


def compare(primary, baseline, seed):
    groups = sorted(next(iter(primary["per_seed"].values()))["groups"]); accuracy, nll = [], []
    for group in groups:
        accuracy.append(statistics.fmean(primary["per_seed"][str(s)]["groups"][group]["accuracy"] -
                                         baseline["per_seed"][str(s)]["groups"][group]["accuracy"] for s in SEEDS))
        nll.append(statistics.fmean(baseline["per_seed"][str(s)]["groups"][group]["nll"] -
                                    primary["per_seed"][str(s)]["groups"][group]["nll"] for s in SEEDS))
    return {"accuracy_gain": statistics.fmean(accuracy), "accuracy_bootstrap_95ci": _bootstrap(accuracy, seed),
            "nll_gain": statistics.fmean(nll), "nll_bootstrap_95ci": _bootstrap(nll, seed + 1)}


def development_gate(summary):
    fixed = ("fixed_retain", "fixed_half", "fixed_local", "fixed_global")
    best_fixed = max(fixed, key=lambda name: summary[name]["mean_accuracy"])
    stronger = max((best_fixed, "global_learned"), key=lambda name: summary[name]["mean_accuracy"])
    learned, oracle = summary["learned"], summary["oracle"]; gain = compare(learned, summary[stronger], 2331)
    checks = {"oracle_readable": oracle["mean_accuracy"] >= .90,
              "learned_accuracy": learned["mean_accuracy"] >= .88,
              "all_seeds": learned["min_seed_accuracy"] >= .85,
              "near_oracle": oracle["mean_accuracy"] - learned["mean_accuracy"] <= .03,
              "control_margin": gain["accuracy_gain"] >= .08,
              "control_confidence": gain["accuracy_bootstrap_95ci"][0] > .05}
    return {"checks": checks, "best_fixed": best_fixed, "stronger_control": stronger,
            "comparison": gain, "passed": all(checks.values())}


class QwenEmbeddings:
    def __init__(self, seed, dim, values):
        self.seed, self.dim, self.values = seed, dim, values

    def get(self, kind, text):
        return self.values[kind, text]


def strings(pairs):
    entities, values = set(), {"unknown"}
    for _, target in pairs:
        for events in target["semantic_events"]:
            for event in events:
                if event["op"] == "assign":
                    entities.update(event["facts"])
                    values.update(event["facts"].values())
    return sorted(entities), sorted(values)


def _mean_rows(weight, tokenizer, texts, prefix):
    import torch
    encoded = {text: tokenizer.encode(prefix + text, add_special_tokens=False) for text in texts}
    if any(not ids for ids in encoded.values()): raise ValueError("empty Qwen embedding string")
    unique = sorted({token for ids in encoded.values() for token in ids})
    index = torch.tensor(unique, device=weight.device)
    rows = weight[index].float().cpu(); lookup = {token: rows[i] for i, token in enumerate(unique)}
    return {text: torch.stack([lookup[token] for token in ids]).mean(0) for text, ids in encoded.items()}


def build_embeddings(model, tokenizer, pairs, dim, transforms=None):
    import torch
    entities, values = strings(pairs)
    raw = {"entity": _mean_rows(model.get_input_embeddings().weight, tokenizer, entities, ""),
           "value": _mean_rows(model.lm_head.weight, tokenizer, values, " ")}
    centered, fitted = {}, transforms is None
    if transforms is None: transforms = {"means": {}, "projections": {}}
    for kind, names in (("entity", entities), ("value", values)):
        mean = torch.stack([raw[kind][name] for name in names]).mean(0) if fitted else transforms["means"][kind]
        if fitted: transforms["means"][kind] = mean
        centered[kind] = {name: raw[kind][name] - mean for name in names}
    output, meta = {}, {"entities": len(entities), "values": len(values), "dim": dim, "seeds": list(SEEDS), "hashes": {}}
    for seed in SEEDS:
        generator = torch.Generator().manual_seed(seed)
        values_by_name = {}
        for kind, names in (("entity", entities), ("value", values)):
            width = next(iter(centered[kind].values())).numel()
            projection = (torch.randn(width, dim, generator=generator, dtype=torch.float32) / width ** .5
                          if fitted else transforms["projections"][str(seed), kind])
            if fitted: transforms["projections"][str(seed), kind] = projection
            for name in names:
                value = centered[kind][name] @ projection
                values_by_name[kind, name] = value / value.norm().clamp_min(1e-8)
        h = hashlib.sha256()
        for key, value in sorted(values_by_name.items()):
            h.update(repr(key).encode()); h.update(value.contiguous().view(torch.uint8).numpy().tobytes())
        output[str(seed)] = QwenEmbeddings(seed, dim, values_by_name); meta["hashes"][str(seed)] = h.hexdigest()
    meta["entity_names_sha256"] = hashlib.sha256(json.dumps(entities).encode()).hexdigest()
    meta["value_names_sha256"] = hashlib.sha256(json.dumps(values).encode()).hexdigest()
    return output, meta, transforms


def run(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_native import load_model

    root, data_root = Path(args.output), Path(args.data_root); root.mkdir(parents=True)
    manifest = json.loads((data_root / "manifest.json").read_text())
    if manifest["group_counts"] != {"train": 128, "dev": 32, "test": 32}: raise ValueError("registered split required")
    train_pairs, dev_pairs = load_split(data_root, "train"), load_split(data_root, "dev")
    # Names are collected without reading any test answers. Test uses disjoint namespaces and
    # is embedded only after the development gate; Qwen token rows themselves are frozen.
    model, model_meta = load_model(args.model, "step81381_native")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    embeddings, embedding_meta, transforms = build_embeddings(model, tokenizer, train_pairs + dev_pairs, args.dim)
    del model; torch.cuda.empty_cache()
    controllers_saved, training, dev_seed = {}, {}, {}; start = time.perf_counter()
    for seed in SEEDS:
        memory = embeddings[str(seed)]
        full, full_meta = build_examples(train_pairs, memory, ACTIONS, True)
        blind, blind_meta = build_examples(train_pairs, memory, ACTIONS, False)
        global_rows, global_meta = build_examples(train_pairs, memory, GLOBAL_ACTIONS, True)
        learned, curve = train_controller(full, ACTIONS, args.seed + seed, args.epochs)
        blind_model, blind_curve = train_controller(blind, ACTIONS, args.seed + seed, args.epochs)
        global_model, global_curve = train_controller(global_rows, GLOBAL_ACTIONS, args.seed + seed, args.epochs)
        controllers = {"learned": learned, "reference_blind": blind_model, "global_learned": global_model}
        controllers_saved[str(seed)] = {name: serialize_controller(value) for name, value in controllers.items()}
        training[str(seed)] = {"full": full_meta, "reference_blind": blind_meta, "global_only": global_meta,
                               "curve": curve, "reference_blind_curve": blind_curve, "global_curve": global_curve}
        dev_seed[str(seed)] = {}
        for policy in POLICIES:
            records = evaluate_policy(dev_pairs, policy, memory, controllers)
            dev_seed[str(seed)][policy] = aggregate(records)
            path = root / f"dev_seed{seed}_{policy}.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in records))
    dev = multi_seed_summary(dev_seed); gate = development_gate(dev)
    test_scored, test_seed, test, test_embedding_meta = False, None, None, None
    if gate["passed"]:
        test_pairs = load_split(data_root, "test")
        # Apply the train/development centering and projections to unseen test names.
        model, _ = load_model(args.model, "step81381_native")
        test_embeddings, test_embedding_meta, _ = build_embeddings(model, tokenizer, test_pairs, args.dim, transforms)
        del model; torch.cuda.empty_cache(); test_seed = {}
        for seed in SEEDS:
            memory = test_embeddings[str(seed)]
            controllers = restore_controllers(controllers_saved[str(seed)], args.dim)
            test_seed[str(seed)] = {}
            for policy in POLICIES:
                records = evaluate_policy(test_pairs, policy, memory, controllers)
                test_seed[str(seed)][policy] = aggregate(records)
                path = root / f"test_seed{seed}_{policy}.jsonl"
                path.write_text("".join(json.dumps(row) + "\n" for row in records))
        test = multi_seed_summary(test_seed); test_scored = True
    checkpoint = root / "controllers.pt"
    torch.save({"controllers": controllers_saved, "dim": args.dim, "seeds": SEEDS,
                "epochs": args.epochs, "seed": args.seed}, checkpoint)
    raw_hashes = {path.name: sha(path) for path in root.glob("*.jsonl")}
    training_result = {"seconds": time.perf_counter() - start, "seeds": training,
                       "checkpoint_sha256": sha(checkpoint), "embedding_meta": embedding_meta,
                       "test_embedding_meta": test_embedding_meta, "raw_rows_sha256": raw_hashes}
    write_json(root / "training.json", training_result)
    run_manifest = {"protocol": "qwen_lowrank_forgetting_v1", "data_manifest_sha256": sha(data_root / "manifest.json"),
                    "group_counts": manifest["group_counts"], "memory_seeds": list(SEEDS), "dim": args.dim,
                    "actions": list(ACTIONS), "global_grid": [0., .5, 1.], "model": args.model,
                    "tokenizer": args.tokenizer, "loading_info": model_meta["loading_info"],
                    "test_release_rule": "score once only after aggregate development gate", "test_scored": test_scored}
    write_json(root / "training_manifest.json", run_manifest)
    result = {"protocol": "qwen_lowrank_forgetting_v1", "dev": dev, "dev_gate": gate,
              "test": test, "test_scored": test_scored, "stage_passed": gate["passed"],
              "training_seconds": training_result["seconds"], "terminal_status": "completed_qwen_lowrank_forgetting"}
    write_json(root / "summary.json", result); return result


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--data-root", required=True); p.add_argument("--output", required=True)
    p.add_argument("--model", required=True); p.add_argument("--tokenizer", required=True); p.add_argument("--dim", type=int, default=128)
    p.add_argument("--epochs", type=int, default=300); p.add_argument("--seed", type=int, default=2321)
    a = p.parse_args(); print(json.dumps(run(a), indent=2))


if __name__ == "__main__": main()
