"""Diagnose and select an interference-resistant Qwen low-rank memory geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from tasks.build_cbf_event_curriculum import replay
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_cbf_lowrank_forgetting import Embeddings, episode_candidates
from tasks.train_cbf_qwen_lowrank_forgetting import QwenEmbeddings, SEEDS, _mean_rows, strings
from tasks.train_cbf_structured_forgetting import boundary_truth, load_split


REGULARIZERS = (1e-4, 1e-3, 1e-2)
LOADS = (1, 2, 4, 8, 16, 32)
VARIANT_ORDER = ("rp128", "rp128_whiten", "rp256_whiten")


def _hash_table(values):
    h = hashlib.sha256()
    for key, value in sorted(values.items()):
        h.update(repr(key).encode())
        h.update(value.contiguous().view(__import__("torch").uint8).numpy().tobytes())
    return h.hexdigest()


def _normalize_rows(rows):
    return rows / rows.norm(dim=1, keepdim=True).clamp_min(1e-8)


def geometry(rows):
    """Summarize anisotropy without retaining the quadratic cosine matrix."""
    import torch

    rows = _normalize_rows(rows.float())
    covariance = rows.T @ rows / max(1, rows.shape[0] - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    positive = eigenvalues[eigenvalues > eigenvalues.max().clamp_min(1e-12) * 1e-8]
    condition = float(positive.max() / positive.min()) if len(positive) else float("inf")
    probabilities = eigenvalues / eigenvalues.sum().clamp_min(1e-12)
    effective_rank = float(torch.exp(-(probabilities * probabilities.clamp_min(1e-12).log()).sum()))
    # A fixed hash-derived sample makes large tables inexpensive and reproducible.
    generator = torch.Generator().manual_seed(8127 + rows.shape[0] + rows.shape[1])
    samples = min(200000, max(1, rows.shape[0] * 64))
    left = torch.randint(rows.shape[0], (samples,), generator=generator)
    right = torch.randint(rows.shape[0] - 1, (samples,), generator=generator)
    right = right + (right >= left)
    cosines = (rows[left] * rows[right]).sum(-1).abs()
    return {
        "count": rows.shape[0], "dim": rows.shape[1],
        "effective_rank": effective_rank, "condition_number": condition,
        "absolute_off_diagonal_cosine": {
            "mean": float(cosines.mean()),
            "p95": float(torch.quantile(cosines, .95)),
            "p99": float(torch.quantile(cosines, .99)),
            "max_sampled": float(cosines.max()),
            "samples": samples,
        },
    }


def build_representations(model, tokenizer, pairs):
    """Fit all registered projections from train+dev names; never reads test."""
    import torch

    entities, values = strings(pairs)
    raw = {
        "entity": _mean_rows(model.get_input_embeddings().weight, tokenizer, entities, ""),
        "value": _mean_rows(model.lm_head.weight, tokenizer, values, " "),
    }
    names = {"entity": entities, "value": values}
    means = {kind: torch.stack([raw[kind][name] for name in names[kind]]).mean(0)
             for kind in ("entity", "value")}
    centered = {kind: torch.stack([raw[kind][name] - means[kind] for name in names[kind]])
                for kind in ("entity", "value")}
    tables, transforms, meta = {}, {"means": means, "dimensions": {}}, {
        "entities": len(entities), "values": len(values), "seeds": list(SEEDS),
        "regularizers": list(REGULARIZERS), "hashes": {}, "geometry": {},
        "entity_names_sha256": hashlib.sha256(json.dumps(entities).encode()).hexdigest(),
        "value_names_sha256": hashlib.sha256(json.dumps(values).encode()).hexdigest(),
    }
    for dim in (128, 256):
        transforms["dimensions"][dim] = {}
        for seed in SEEDS:
            generator = torch.Generator().manual_seed(seed)
            transforms["dimensions"][dim][seed] = {}
            projected = {}
            for kind in ("entity", "value"):
                width = centered[kind].shape[1]
                projection = torch.randn(width, dim, generator=generator) / width ** .5
                matrix = centered[kind] @ projection
                covariance = matrix.T @ matrix / max(1, matrix.shape[0] - 1)
                eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
                eigenvalues = eigenvalues.clamp_min(0)
                scale = float(eigenvalues.mean().clamp_min(1e-12))
                projected[kind] = matrix
                transforms["dimensions"][dim][seed][kind] = {
                    "projection": projection, "eigenvalues": eigenvalues,
                    "eigenvectors": eigenvectors, "regularizer_scale": scale,
                }
            variants = [None] if dim == 128 else []
            variants += list(REGULARIZERS)
            for regularizer in variants:
                label = ("rp128" if regularizer is None else
                         f"rp{dim}_whiten_r{regularizer:g}")
                values_by_name = {}
                geometry_by_kind = {}
                for kind in ("entity", "value"):
                    matrix = projected[kind]
                    if regularizer is not None:
                        item = transforms["dimensions"][dim][seed][kind]
                        inverse = torch.rsqrt(item["eigenvalues"] + regularizer * item["regularizer_scale"])
                        matrix = (matrix @ item["eigenvectors"] * inverse) @ item["eigenvectors"].T
                    matrix = _normalize_rows(matrix)
                    geometry_by_kind[kind] = geometry(matrix)
                    for index, name in enumerate(names[kind]):
                        values_by_name[kind, name] = matrix[index]
                tables[label, str(seed)] = QwenEmbeddings(seed, dim, values_by_name)
                meta["hashes"].setdefault(label, {})[str(seed)] = _hash_table(values_by_name)
                meta["geometry"].setdefault(label, {})[str(seed)] = geometry_by_kind
    return tables, transforms, meta


def read_metrics(output, answer, candidates, candidate_vectors,
                 temperature=.2, unknown_raw=.35):
    import torch

    raw = [unknown_raw if name == "unknown" else float(torch.dot(value, output))
           for name, value in zip(candidates, candidate_vectors)]
    logits = torch.tensor(raw) / temperature
    truth = candidates.index(answer)
    order = sorted(range(len(candidates)), key=lambda index: (-raw[index], index))
    best_wrong = max(score for index, score in enumerate(raw) if index != truth)
    return {
        "correct": int(order[0] == truth), "prediction": candidates[order[0]],
        "rank": order.index(truth) + 1, "mrr": 1. / (order.index(truth) + 1),
        "margin": raw[truth] - best_wrong,
        "nll": float(-torch.log_softmax(logits, -1)[truth]),
    }


def _read_output(active, query_key, embeddings):
    import torch

    output = torch.zeros(embeddings.dim, dtype=torch.float32)
    for key, value in active.items():
        coefficient = torch.dot(embeddings.get("entity", key), query_key)
        output += coefficient * embeddings.get("value", value)
    return output


def perfect_state_records(pairs, embeddings, variant):
    records = []
    for inp, target in pairs:
        events, queries = [], []
        candidates = episode_candidates(target)
        boundaries = set(inp["decision_boundaries"])
        for block, semantic in zip(inp["blocks"], target["semantic_events"]):
            events.extend(semantic)
            if block["index"] not in boundaries:
                continue
            active = replay(events)
            for query in boundary_truth(inp, target, block["index"]):
                metrics = read_metrics_with_embeddings(active, query, candidates, embeddings)
                queries.append({**query, **metrics, "after_block": block["index"],
                                "active_components": len(active)})
        records.append({"episode_id": inp["id"], "group_id": inp["group_id"],
                        "family": target["family"], "variant": variant,
                        "seed": embeddings.seed, "queries": queries})
    return records


def read_metrics_with_embeddings(active, query, candidates, embeddings):
    values = [embeddings.get("value", value) for value in candidates]
    output = _read_output(active, embeddings.get("entity", query["key"]), embeddings)
    return read_metrics(output, query["answer"], candidates, values)


def aggregate_readability(records):
    pairs = [(record, query) for record in records for query in record["queries"]]
    def metrics(rows):
        return {
            "queries": len(rows),
            "accuracy": statistics.fmean(query["correct"] for _, query in rows),
            "nll": statistics.fmean(query["nll"] for _, query in rows),
            "mrr": statistics.fmean(query["mrr"] for _, query in rows),
            "margin": statistics.fmean(query["margin"] for _, query in rows),
        }
    result = metrics(pairs)
    result["episodes"] = len(records)
    result["groups"] = {}
    for group in sorted({record["group_id"] for record in records}):
        result["groups"][group] = metrics([(r, q) for r, q in pairs if r["group_id"] == group])
    result["by_family"] = {}
    for family in sorted({record["family"] for record in records}):
        result["by_family"][family] = metrics([(r, q) for r, q in pairs if r["family"] == family])
    result["by_active_components"] = {}
    for load in sorted({query["active_components"] for _, query in pairs}):
        result["by_active_components"][str(load)] = metrics(
            [(r, q) for r, q in pairs if q["active_components"] == load])
    return result


def assignment_pairs(pairs):
    mapping = {}
    for _, target in pairs:
        for events in target["semantic_events"]:
            for event in events:
                if event["op"] == "assign":
                    for key, value in event["facts"].items():
                        mapping.setdefault(key, value)
    return sorted(mapping.items())


def load_sweep(pairs, embeddings, trials=128):
    import random

    pool = assignment_pairs(pairs)
    output = {}
    for load in LOADS:
        rows = []
        for trial in range(trials):
            rng = random.Random(4401 + trial)
            selected = rng.sample(pool, load)
            active = dict(selected)
            candidates = sorted({"unknown", *(value for _, value in selected)})
            for key, answer in selected:
                rows.append(read_metrics_with_embeddings(active, {"key": key, "answer": answer},
                                                         candidates, embeddings))
        output[str(load)] = {
            "queries": len(rows), "accuracy": statistics.fmean(row["correct"] for row in rows),
            "mrr": statistics.fmean(row["mrr"] for row in rows),
            "margin": statistics.fmean(row["margin"] for row in rows),
        }
    return output


def mean_seed_summary(seed_metrics):
    return {
        "mean_accuracy": statistics.fmean(row["accuracy"] for row in seed_metrics.values()),
        "min_seed_accuracy": min(row["accuracy"] for row in seed_metrics.values()),
        "mean_nll": statistics.fmean(row["nll"] for row in seed_metrics.values()),
        "mean_mrr": statistics.fmean(row["mrr"] for row in seed_metrics.values()),
        "mean_margin": statistics.fmean(row["margin"] for row in seed_metrics.values()),
        "min_family_accuracy": min(item["accuracy"] for row in seed_metrics.values()
                                   for item in row["by_family"].values()),
        "per_seed": seed_metrics,
    }


def select_regularizer(train_grid, dim):
    prefix = f"rp{dim}_whiten_r"
    candidates = [(float(label.removeprefix(prefix)), row["mean_accuracy"])
                  for label, row in train_grid.items() if label.startswith(prefix)]
    best = max(score for _, score in candidates)
    return max(regularizer for regularizer, score in candidates if score >= best - .01)


def readability_gate(dev, load_sweeps):
    control = dev["random128"]["mean_accuracy"] >= .95
    variants = {}
    for label in VARIANT_ORDER:
        row = dev[label]
        load16 = statistics.fmean(load_sweeps[label][str(seed)]["16"]["accuracy"] for seed in SEEDS)
        checks = {
            "mean_accuracy": row["mean_accuracy"] >= .90,
            "min_seed_accuracy": row["min_seed_accuracy"] >= .85,
            "min_family_accuracy": row["min_family_accuracy"] >= .85,
            "load16_accuracy": load16 >= .85,
            "random_control": control,
        }
        variants[label] = {"checks": checks, "load16_accuracy": load16,
                           "passed": all(checks.values())}
    selected = next((label for label in VARIANT_ORDER if variants[label]["passed"]), None)
    return {"random_control_passed": control, "variants": variants,
            "selected_variant": selected, "passed": selected is not None}


def run(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_native import load_model

    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    train_pairs = load_split(args.data_root, "train")
    dev_pairs = load_split(args.data_root, "dev")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model, model_meta = load_model(args.model, "step81381_native")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tables, transforms, embedding_meta = build_representations(
        model, tokenizer, train_pairs + dev_pairs)
    peak_allocated = torch.cuda.max_memory_allocated() / 2**30
    peak_reserved = torch.cuda.max_memory_reserved() / 2**30
    del model; torch.cuda.empty_cache()

    baseline_bridge = None
    if args.v1_training:
        prior = json.loads(Path(args.v1_training).read_text())
        expected = prior["embedding_meta"]["hashes"]
        observed = embedding_meta["hashes"]["rp128"]
        baseline_bridge = {"expected": expected, "observed": observed, "exact": expected == observed}
        if not baseline_bridge["exact"]:
            raise ValueError("rp128 does not reproduce qwen_lowrank_forgetting_v1")

    train_grid = {}
    for label, seed in sorted(tables):
        records = perfect_state_records(train_pairs, tables[label, seed], label)
        train_grid.setdefault(label, {})[seed] = aggregate_readability(records)
    train_summary = {label: mean_seed_summary(rows) for label, rows in train_grid.items()}
    selected_regs = {str(dim): select_regularizer(train_summary, dim) for dim in (128, 256)}
    chosen = {
        "rp128": "rp128",
        "rp128_whiten": f"rp128_whiten_r{selected_regs['128']:g}",
        "rp256_whiten": f"rp256_whiten_r{selected_regs['256']:g}",
    }

    dev, load_sweeps, raw_hashes = {}, {}, {}
    candidates = {**chosen, "random128": "random128"}
    for label, internal in candidates.items():
        dev_seed, load_sweeps[label] = {}, {}
        for seed in SEEDS:
            embeddings = Embeddings(seed, 128) if label == "random128" else tables[internal, str(seed)]
            records = perfect_state_records(dev_pairs, embeddings, label)
            path = root / f"dev_{label}_seed{seed}.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in records))
            raw_hashes[path.name] = sha(path)
            dev_seed[str(seed)] = aggregate_readability(records)
            load_sweeps[label][str(seed)] = load_sweep(dev_pairs, embeddings, args.load_trials)
        dev[label] = mean_seed_summary(dev_seed)
    gate = readability_gate(dev, load_sweeps)

    transform_path = root / "transforms.pt"
    torch.save({"protocol": "qwen_lowrank_geometry_v1", "transforms": transforms,
                "selected_regularizers": selected_regs, "chosen": chosen,
                "embedding_hashes": embedding_meta["hashes"]}, transform_path)
    training = {
        "seconds": time.perf_counter() - started,
        "gpu_peak_allocated_gib": peak_allocated,
        "gpu_peak_reserved_gib": peak_reserved,
        "embedding_meta": embedding_meta,
        "train_grid": train_summary,
        "selected_regularizers": selected_regs,
        "chosen_internal_variants": chosen,
        "load_sweeps": load_sweeps,
        "raw_rows_sha256": raw_hashes,
        "transforms_sha256": sha(transform_path),
        "baseline_bridge": baseline_bridge,
    }
    manifest = {
        "protocol": "qwen_lowrank_geometry_v1",
        "data_manifest_sha256": sha(Path(args.data_root) / "manifest.json"),
        "model": args.model, "tokenizer": args.tokenizer,
        "loading_info": model_meta["loading_info"], "seeds": list(SEEDS),
        "registered_variants": list(VARIANT_ORDER),
        "regularizers": list(REGULARIZERS), "loads": list(LOADS),
        "fit_splits": ["train", "dev"], "selection_split": "train",
        "gate_split": "dev", "test_loaded": False, "test_scored": False,
    }
    result = {
        "protocol": "qwen_lowrank_geometry_v1",
        "selected_regularizers": selected_regs,
        "chosen_internal_variants": chosen,
        "dev": dev, "readability_gate": gate,
        "selected_variant": gate["selected_variant"],
        "stage_passed": gate["passed"], "test": None, "test_scored": False,
        "training_seconds": training["seconds"],
        "terminal_status": "completed_qwen_lowrank_geometry",
    }
    write_json(root / "training.json", training)
    write_json(root / "training_manifest.json", manifest)
    write_json(root / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--v1-training")
    parser.add_argument("--load-trials", type=int, default=128)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
