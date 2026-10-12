"""Diagnose contextual Qwen key/value geometry after mean-token representations fail."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_cbf_lowrank_forgetting import Embeddings
from tasks.train_cbf_qwen_lowrank_forgetting import QwenEmbeddings, SEEDS, strings
from tasks.train_cbf_structured_forgetting import load_split
from tasks.diagnose_cbf_qwen_lowrank_geometry import (
    REGULARIZERS, aggregate_readability, build_representations, geometry,
    load_sweep, mean_seed_summary, perfect_state_records,
)


UNKNOWN_GRID = (.1, .2, .35, .5, .75)
POOLINGS = ("context_last", "context_mean")
DEV_VARIANTS = (
    "mean_qwen", "mean_qkey_random_value", "random_key_mean_qvalue",
    "random256", "context_last", "context_mean",
)
PROMPTS = {"entity": "Entity identifier: ", "value": "Assigned value code: "}


class HybridEmbeddings:
    def __init__(self, key_source, value_source):
        if key_source.seed != value_source.seed or key_source.dim != value_source.dim:
            raise ValueError("hybrid representation mismatch")
        self.seed, self.dim = key_source.seed, key_source.dim
        self.key_source, self.value_source = key_source, value_source

    def get(self, kind, text):
        source = self.key_source if kind == "entity" else self.value_source
        return source.get(kind, text)


def _hash_table(values):
    import torch

    h = hashlib.sha256()
    for key, value in sorted(values.items()):
        h.update(repr(key).encode())
        h.update(value.contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def extract_contextual(model, tokenizer, pairs, batch_size):
    """Extract final-layer target-span states with an explicit token boundary."""
    import torch

    entities, values = strings(pairs)
    names = {"entity": entities, "value": values}
    raw = {pooling: {kind: {} for kind in names} for pooling in POOLINGS}
    prompt_ids = {kind: tokenizer.encode(prompt, add_special_tokens=False)
                  for kind, prompt in PROMPTS.items()}
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id
    for kind in ("entity", "value"):
        for offset in range(0, len(names[kind]), batch_size):
            batch = names[kind][offset:offset + batch_size]
            target_ids = [tokenizer.encode(text, add_special_tokens=False) for text in batch]
            if any(not ids for ids in target_ids):
                raise ValueError("empty contextual identifier")
            rows = [prompt_ids[kind] + ids for ids in target_ids]
            width = max(map(len, rows))
            input_ids = torch.full((len(rows), width), pad, dtype=torch.long, device="cuda")
            attention = torch.zeros_like(input_ids)
            for index, ids in enumerate(rows):
                input_ids[index, :len(ids)] = torch.tensor(ids, device="cuda")
                attention[index, :len(ids)] = 1
            with torch.inference_mode():
                hidden = model.model(input_ids=input_ids, attention_mask=attention,
                                     use_cache=False, return_dict=True).last_hidden_state.float().cpu()
            start = len(prompt_ids[kind])
            for index, (text, ids) in enumerate(zip(batch, target_ids)):
                span = hidden[index, start:start + len(ids)]
                raw["context_last"][kind][text] = span[-1]
                raw["context_mean"][kind][text] = span.mean(0)
    meta = {
        "prompts": PROMPTS, "prompt_token_ids": prompt_ids,
        "entities": len(entities), "values": len(values),
        "entity_names_sha256": hashlib.sha256(json.dumps(entities).encode()).hexdigest(),
        "value_names_sha256": hashlib.sha256(json.dumps(values).encode()).hexdigest(),
    }
    return raw, names, meta


def project_contextual(raw, names, dim=256):
    import torch

    tables, transforms = {}, {}
    meta = {"hashes": {}, "geometry": {}, "dim": dim, "seeds": list(SEEDS)}
    for pooling in POOLINGS:
        means = {kind: torch.stack([raw[pooling][kind][name] for name in names[kind]]).mean(0)
                 for kind in ("entity", "value")}
        centered = {kind: torch.stack([raw[pooling][kind][name] - means[kind]
                                       for name in names[kind]])
                    for kind in ("entity", "value")}
        transforms[pooling] = {"means": means, "seeds": {}}
        for seed in SEEDS:
            generator = torch.Generator().manual_seed(seed)
            projected, transforms[pooling]["seeds"][seed] = {}, {}
            for kind in ("entity", "value"):
                width = centered[kind].shape[1]
                projection = torch.randn(width, dim, generator=generator) / width ** .5
                matrix = centered[kind] @ projection
                covariance = matrix.T @ matrix / max(1, matrix.shape[0] - 1)
                eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
                eigenvalues = eigenvalues.clamp_min(0)
                scale = float(eigenvalues.mean().clamp_min(1e-12))
                projected[kind] = matrix
                transforms[pooling]["seeds"][seed][kind] = {
                    "projection": projection, "eigenvalues": eigenvalues,
                    "eigenvectors": eigenvectors, "regularizer_scale": scale,
                }
            for regularizer in REGULARIZERS:
                label = f"{pooling}_r{regularizer:g}"
                values_by_name, by_kind = {}, {}
                for kind in ("entity", "value"):
                    item = transforms[pooling]["seeds"][seed][kind]
                    inverse = torch.rsqrt(item["eigenvalues"] + regularizer * item["regularizer_scale"])
                    matrix = (projected[kind] @ item["eigenvectors"] * inverse) @ item["eigenvectors"].T
                    matrix = matrix / matrix.norm(dim=1, keepdim=True).clamp_min(1e-8)
                    by_kind[kind] = geometry(matrix)
                    for index, name in enumerate(names[kind]):
                        values_by_name[kind, name] = matrix[index]
                tables[label, str(seed)] = QwenEmbeddings(seed, dim, values_by_name)
                meta["hashes"].setdefault(label, {})[str(seed)] = _hash_table(values_by_name)
                meta["geometry"].setdefault(label, {})[str(seed)] = by_kind
    return tables, transforms, meta


def apply_contextual_transform(raw, names, transforms, pooling, regularizer, dim=256):
    """Apply a frozen train+dev contextual transform to any disjoint namespace."""
    import torch

    output, hashes = {}, {}
    for seed in SEEDS:
        values_by_name = {}
        for kind in ("entity", "value"):
            item = transforms[pooling]["seeds"][seed][kind]
            centered = torch.stack([
                raw[pooling][kind][name] - transforms[pooling]["means"][kind]
                for name in names[kind]
            ])
            projected = centered @ item["projection"]
            inverse = torch.rsqrt(item["eigenvalues"] + regularizer * item["regularizer_scale"])
            matrix = (projected @ item["eigenvectors"] * inverse) @ item["eigenvectors"].T
            matrix = matrix / matrix.norm(dim=1, keepdim=True).clamp_min(1e-8)
            for index, name in enumerate(names[kind]):
                values_by_name[kind, name] = matrix[index]
        output[str(seed)] = QwenEmbeddings(seed, dim, values_by_name)
        hashes[str(seed)] = _hash_table(values_by_name)
    return output, hashes


def train_grid(pairs, tables):
    output = {}
    for (internal, seed), embeddings in sorted(tables.items()):
        output.setdefault(internal, {})
        for threshold in UNKNOWN_GRID:
            records = perfect_state_records(pairs, embeddings, internal, unknown_raw=threshold)
            output[internal].setdefault(str(threshold), {})[seed] = \
                aggregate_readability(records)["accuracy"]
    return output


def select_context_config(grid, pooling):
    candidates = []
    for regularizer in REGULARIZERS:
        internal = f"{pooling}_r{regularizer:g}"
        thresholds = []
        for threshold in UNKNOWN_GRID:
            values = grid[internal][str(threshold)]
            thresholds.append((statistics.fmean(values.values()), threshold))
        best_accuracy = max(score for score, _ in thresholds)
        best_threshold = min((threshold for score, threshold in thresholds if score == best_accuracy),
                             key=lambda value: (abs(value - .35), value))
        candidates.append({"internal": internal, "regularizer": regularizer,
                           "unknown_raw": best_threshold, "train_accuracy": best_accuracy})
    best = max(row["train_accuracy"] for row in candidates)
    return max((row for row in candidates if row["train_accuracy"] >= best - .01),
               key=lambda row: row["regularizer"])


def contextual_gate(dev, sweeps):
    control = dev["random256"]["mean_accuracy"] >= .95
    variants = {}
    for label in POOLINGS:
        row = dev[label]
        load16 = statistics.fmean(sweeps[label][str(seed)]["16"]["accuracy"] for seed in SEEDS)
        checks = {
            "mean_accuracy": row["mean_accuracy"] >= .90,
            "min_seed_accuracy": row["min_seed_accuracy"] >= .85,
            "min_family_accuracy": row["min_family_accuracy"] >= .85,
            "load16_accuracy": load16 >= .85,
            "random_control": control,
        }
        variants[label] = {"checks": checks, "load16_accuracy": load16,
                           "passed": all(checks.values())}
    selected = next((label for label in POOLINGS if variants[label]["passed"]), None)
    return {"random_control_passed": control, "variants": variants,
            "selected_variant": selected, "passed": selected is not None}


def run(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_native import load_model

    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    train_pairs = load_split(args.data_root, "train")
    dev_pairs = load_split(args.data_root, "dev")
    source_summary = json.loads(Path(args.geometry_summary).read_text())
    source_training = json.loads(Path(args.geometry_training).read_text())
    if source_summary["protocol"] != "qwen_lowrank_geometry_v1" or source_summary["stage_passed"]:
        raise ValueError("contextual stage requires the audited failed mean-token geometry")
    if source_training["selected_regularizers"]["256"] != .01:
        raise ValueError("unexpected source regularizer")

    torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    model, model_meta = load_model(args.model, "step81381_native")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    mean_tables, _, mean_meta = build_representations(model, tokenizer, train_pairs + dev_pairs)
    internal_mean = "rp256_whiten_r0.01"
    source_hashes = source_training["embedding_meta"]["hashes"][internal_mean]
    if mean_meta["hashes"][internal_mean] != source_hashes:
        raise ValueError("mean-token source bridge mismatch")
    raw, names, extraction_meta = extract_contextual(
        model, tokenizer, train_pairs + dev_pairs, args.batch_size)
    context_tables, transforms, context_meta = project_contextual(raw, names)
    peak_allocated = torch.cuda.max_memory_allocated() / 2**30
    peak_reserved = torch.cuda.max_memory_reserved() / 2**30
    del model, raw; torch.cuda.empty_cache()

    grid = train_grid(train_pairs, context_tables)
    selected = {pooling: select_context_config(grid, pooling) for pooling in POOLINGS}
    dev, sweeps, raw_hashes = {}, {}, {}
    for label in DEV_VARIANTS:
        per_seed, sweeps[label] = {}, {}
        for seed in SEEDS:
            random = Embeddings(seed, 256)
            qwen = mean_tables[internal_mean, str(seed)]
            if label == "mean_qwen":
                embeddings, threshold = qwen, .35
            elif label == "mean_qkey_random_value":
                embeddings, threshold = HybridEmbeddings(qwen, random), .35
            elif label == "random_key_mean_qvalue":
                embeddings, threshold = HybridEmbeddings(random, qwen), .35
            elif label == "random256":
                embeddings, threshold = random, .35
            else:
                config = selected[label]
                embeddings = context_tables[config["internal"], str(seed)]
                threshold = config["unknown_raw"]
            records = perfect_state_records(dev_pairs, embeddings, label, unknown_raw=threshold)
            path = root / f"dev_{label}_seed{seed}.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in records))
            raw_hashes[path.name] = sha(path)
            per_seed[str(seed)] = aggregate_readability(records)
            sweeps[label][str(seed)] = load_sweep(dev_pairs, embeddings, args.load_trials)
        dev[label] = mean_seed_summary(per_seed)
    gate = contextual_gate(dev, sweeps)

    transform_path = root / "contextual_transforms.pt"
    torch.save({"protocol": "qwen_contextual_geometry_v1", "transforms": transforms,
                "selected": selected, "embedding_hashes": context_meta["hashes"],
                "extraction": extraction_meta}, transform_path)
    training = {
        "seconds": time.perf_counter() - started,
        "gpu_peak_allocated_gib": peak_allocated,
        "gpu_peak_reserved_gib": peak_reserved,
        "source_geometry_summary_sha256": sha(args.geometry_summary),
        "source_geometry_training_sha256": sha(args.geometry_training),
        "source_mean_embedding_hashes": source_hashes,
        "mean_embedding_bridge_exact": True,
        "extraction_meta": extraction_meta, "context_embedding_meta": context_meta,
        "train_grid": grid, "selected_configs": selected, "load_sweeps": sweeps,
        "raw_rows_sha256": raw_hashes,
        "transforms_sha256": sha(transform_path),
    }
    manifest = {
        "protocol": "qwen_contextual_geometry_v1",
        "data_manifest_sha256": sha(Path(args.data_root) / "manifest.json"),
        "model": args.model, "tokenizer": args.tokenizer,
        "loading_info": model_meta["loading_info"], "seeds": list(SEEDS),
        "poolings": list(POOLINGS), "regularizers": list(REGULARIZERS),
        "unknown_grid": list(UNKNOWN_GRID), "dev_variants": list(DEV_VARIANTS),
        "fit_splits": ["train", "dev"], "selection_split": "train",
        "gate_split": "dev", "test_loaded": False, "test_scored": False,
    }
    result = {
        "protocol": "qwen_contextual_geometry_v1", "selected_configs": selected,
        "dev": dev, "contextual_gate": gate,
        "selected_variant": gate["selected_variant"],
        "stage_passed": gate["passed"], "test": None, "test_scored": False,
        "training_seconds": training["seconds"],
        "terminal_status": "completed_qwen_contextual_geometry",
    }
    write_json(root / "training.json", training)
    write_json(root / "training_manifest.json", manifest)
    write_json(root / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--geometry-summary", required=True)
    parser.add_argument("--geometry-training", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--load-trials", type=int, default=128)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
