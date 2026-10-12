"""Evaluate natural ZsRE prompt/rephrase readability of contextual low-rank memory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_cbf_qwen_lowrank_forgetting import QwenEmbeddings
from tasks.diagnose_cbf_qwen_lowrank_geometry import read_metrics_with_embeddings


SEEDS = (5101, 5102, 5103)
REGULARIZERS = (1e-4, 1e-3, 1e-2)
UNKNOWN_GRID = (.1, .2, .35, .5, .75)
PREFIXES = {"entity": "Memory key: ", "value": "Memory value: "}
QUERY_TYPES = (
    "original_exact", "original_rephrase", "original_locality",
    "updated_exact", "updated_rephrase", "updated_locality",
)


def load_split(root, split):
    root = Path(root); manifest = json.loads((root / "manifest.json").read_text())
    if manifest["protocol"] != "zsre_contextual_readout_v1":
        raise ValueError("ZsRE benchmark protocol mismatch")
    path = root / f"{split}.jsonl"
    if sha(path) != manifest["files_sha256"][path.name]:
        raise ValueError("ZsRE split hash mismatch")
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def texts(sessions):
    keys, values = set(), {"unknown"}
    for session in sessions:
        for fact in session["facts"]:
            keys.update((fact["prompt"], fact["rephrase_prompt"], *fact["locality_prompts"]))
            values.update((fact["original"], fact["updated"]))
    return {"entity": sorted(keys), "value": sorted(values)}


def extract(model, tokenizer, names, batch_size):
    import torch

    raw = {kind: {} for kind in names}
    prefix_ids = {kind: tokenizer.encode(prefix, add_special_tokens=False)
                  for kind, prefix in PREFIXES.items()}
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    for kind in ("entity", "value"):
        for offset in range(0, len(names[kind]), batch_size):
            batch = names[kind][offset:offset + batch_size]
            target_ids = [tokenizer.encode(text, add_special_tokens=False) for text in batch]
            if any(not row for row in target_ids):
                raise ValueError("empty ZsRE tokenization")
            rows = [prefix_ids[kind] + row for row in target_ids]
            width = max(map(len, rows))
            ids = torch.full((len(rows), width), pad, dtype=torch.long, device="cuda")
            mask = torch.zeros_like(ids)
            for index, row in enumerate(rows):
                ids[index, :len(row)] = torch.tensor(row, device="cuda"); mask[index, :len(row)] = 1
            with torch.inference_mode():
                hidden = model.model(input_ids=ids, attention_mask=mask, use_cache=False,
                                     return_dict=True).last_hidden_state.float().cpu()
            for index, (name, row) in enumerate(zip(batch, target_ids)):
                raw[kind][name] = hidden[index, len(prefix_ids[kind]) + len(row) - 1]
    return raw


def _hash_table(values):
    import torch

    h = hashlib.sha256()
    for key, value in sorted(values.items()):
        h.update(repr(key).encode()); h.update(value.contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def fit(raw, names, dim=256):
    import torch

    means = {kind: torch.stack([raw[kind][name] for name in names[kind]]).mean(0)
             for kind in ("entity", "value")}
    centered = {kind: torch.stack([raw[kind][name] - means[kind] for name in names[kind]])
                for kind in ("entity", "value")}
    transforms, tables, hashes = {"means": means, "seeds": {}}, {}, {}
    for seed in SEEDS:
        generator = torch.Generator().manual_seed(seed); transforms["seeds"][seed] = {}
        projected = {}
        for kind in ("entity", "value"):
            width = centered[kind].shape[1]
            projection = torch.randn(width, dim, generator=generator) / width ** .5
            matrix = centered[kind] @ projection
            covariance = matrix.T @ matrix / max(1, matrix.shape[0] - 1)
            eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
            eigenvalues = eigenvalues.clamp_min(0); scale = float(eigenvalues.mean().clamp_min(1e-12))
            projected[kind] = matrix
            transforms["seeds"][seed][kind] = {
                "projection": projection, "eigenvalues": eigenvalues,
                "eigenvectors": eigenvectors, "regularizer_scale": scale,
            }
        for regularizer in REGULARIZERS:
            label = f"context_last_r{regularizer:g}"; values_by_name = {}
            for kind in ("entity", "value"):
                item = transforms["seeds"][seed][kind]
                inverse = torch.rsqrt(item["eigenvalues"] + regularizer * item["regularizer_scale"])
                matrix = (projected[kind] @ item["eigenvectors"] * inverse) @ item["eigenvectors"].T
                matrix = matrix / matrix.norm(dim=1, keepdim=True).clamp_min(1e-8)
                for index, name in enumerate(names[kind]): values_by_name[kind, name] = matrix[index]
            tables[label, str(seed)] = QwenEmbeddings(seed, dim, values_by_name)
            hashes.setdefault(label, {})[str(seed)] = _hash_table(values_by_name)
    return tables, transforms, hashes


def apply(raw, names, transforms, regularizer, dim=256):
    import torch

    output, hashes = {}, {}
    for seed in SEEDS:
        values_by_name = {}
        for kind in ("entity", "value"):
            item = transforms["seeds"][seed][kind]
            centered = torch.stack([raw[kind][name] - transforms["means"][kind]
                                    for name in names[kind]])
            projected = centered @ item["projection"]
            inverse = torch.rsqrt(item["eigenvalues"] + regularizer * item["regularizer_scale"])
            matrix = (projected @ item["eigenvectors"] * inverse) @ item["eigenvectors"].T
            matrix = matrix / matrix.norm(dim=1, keepdim=True).clamp_min(1e-8)
            for index, name in enumerate(names[kind]): values_by_name[kind, name] = matrix[index]
        output[str(seed)] = QwenEmbeddings(seed, dim, values_by_name)
        hashes[str(seed)] = _hash_table(values_by_name)
    return output, hashes


def evaluate(sessions, embeddings, unknown_raw, split):
    records = []
    for session in sessions:
        candidates = sorted({"unknown", *(fact[field] for fact in session["facts"]
                                           for field in ("original", "updated"))})
        queries = []
        for memory_name, field in (("original", "original"), ("updated", "updated")):
            active = {fact["prompt"]: fact[field] for fact in session["facts"]}
            for fact in session["facts"]:
                for query_name, key, answer in (
                    (f"{memory_name}_exact", fact["prompt"], fact[field]),
                    (f"{memory_name}_rephrase", fact["rephrase_prompt"], fact[field]),
                    (f"{memory_name}_locality", fact["locality_prompts"][0], "unknown"),
                    (f"{memory_name}_locality", fact["locality_prompts"][1], "unknown"),
                ):
                    metrics = read_metrics_with_embeddings(
                        active, {"key": key, "answer": answer}, candidates, embeddings,
                        unknown_raw=unknown_raw)
                    queries.append({"subject": fact["subject"], "type": query_name,
                                    "key": key, "answer": answer, **metrics})
        records.append({"session_id": session["id"], "split": split,
                        "seed": embeddings.seed, "queries": queries})
    return records


def aggregate(records):
    queries = [(record, query) for record in records for query in record["queries"]]
    def metrics(rows):
        return {"queries": len(rows),
                "accuracy": statistics.fmean(query["correct"] for _, query in rows),
                "nll": statistics.fmean(query["nll"] for _, query in rows),
                "mrr": statistics.fmean(query["mrr"] for _, query in rows),
                "margin": statistics.fmean(query["margin"] for _, query in rows)}
    result = metrics(queries); result["sessions"] = len(records)
    result["by_type"] = {kind: metrics([(r, q) for r, q in queries if q["type"] == kind])
                         for kind in QUERY_TYPES}
    result["groups"] = {record["session_id"]: metrics(
        [(r, q) for r, q in queries if r["session_id"] == record["session_id"]])
        for record in records}
    return result


def summarize(per_seed):
    return {"mean_accuracy": statistics.fmean(row["accuracy"] for row in per_seed.values()),
            "min_seed_accuracy": min(row["accuracy"] for row in per_seed.values()),
            "mean_nll": statistics.fmean(row["nll"] for row in per_seed.values()),
            "by_type": {kind: {"accuracy": statistics.fmean(
                row["by_type"][kind]["accuracy"] for row in per_seed.values()),
                "nll": statistics.fmean(row["by_type"][kind]["nll"] for row in per_seed.values())}
                for kind in QUERY_TYPES}, "per_seed": per_seed}


def select_config(grid):
    candidates = []
    for regularizer in REGULARIZERS:
        label = f"context_last_r{regularizer:g}"; thresholds = []
        for threshold in UNKNOWN_GRID:
            scores = grid[label][str(threshold)]
            thresholds.append((statistics.fmean(scores.values()), threshold))
        best = max(score for score, _ in thresholds)
        threshold = min((value for score, value in thresholds if score == best),
                        key=lambda value: (abs(value - .35), value))
        candidates.append({"internal": label, "regularizer": regularizer,
                           "unknown_raw": threshold, "calibration_accuracy": best})
    best = max(row["calibration_accuracy"] for row in candidates)
    return max((row for row in candidates if row["calibration_accuracy"] >= best - .01),
               key=lambda row: row["regularizer"])


def gate(summary):
    by_type = summary["by_type"]
    locality = statistics.fmean(by_type[kind]["accuracy"] for kind in
                                ("original_locality", "updated_locality"))
    checks = {
        "original_exact": by_type["original_exact"]["accuracy"] >= .90,
        "updated_exact": by_type["updated_exact"]["accuracy"] >= .90,
        "original_rephrase": by_type["original_rephrase"]["accuracy"] >= .70,
        "updated_rephrase": by_type["updated_rephrase"]["accuracy"] >= .70,
        "locality_unknown": locality >= .90,
        "min_seed_overall": summary["min_seed_accuracy"] >= .75,
    }
    return {"checks": checks, "locality_accuracy": locality, "passed": all(checks.values())}


def run(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_native import load_model

    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    calibration = load_split(args.data_root, "calibration")
    dev_sessions = load_split(args.data_root, "dev")
    torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    model, model_meta = load_model(args.model, "step81381_native")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    calibration_names = texts(calibration)
    calibration_raw = extract(model, tokenizer, calibration_names, args.batch_size)
    tables, transforms, calibration_hashes = fit(calibration_raw, calibration_names)
    grid = {}
    for (label, seed), embeddings in sorted(tables.items()):
        grid.setdefault(label, {})
        for threshold in UNKNOWN_GRID:
            records = evaluate(calibration, embeddings, threshold, "calibration")
            grid[label].setdefault(str(threshold), {})[seed] = aggregate(records)["accuracy"]
    selected = select_config(grid)
    dev_names = texts(dev_sessions); dev_raw = extract(model, tokenizer, dev_names, args.batch_size)
    dev_embeddings, dev_hashes = apply(
        dev_raw, dev_names, transforms, selected["regularizer"])
    dev_seed, raw_hashes = {}, {}
    for seed in SEEDS:
        records = evaluate(dev_sessions, dev_embeddings[str(seed)], selected["unknown_raw"], "dev")
        path = root / f"dev_seed{seed}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
        raw_hashes[path.name] = sha(path); dev_seed[str(seed)] = aggregate(records)
    dev = summarize(dev_seed); dev_gate = gate(dev)

    test_scored, test, test_hashes = False, None, None
    if dev_gate["passed"]:
        test_sessions = load_split(args.data_root, "test")
        test_names = texts(test_sessions); test_raw = extract(model, tokenizer, test_names, args.batch_size)
        test_embeddings, test_hashes = apply(
            test_raw, test_names, transforms, selected["regularizer"])
        test_seed = {}
        for seed in SEEDS:
            records = evaluate(test_sessions, test_embeddings[str(seed)], selected["unknown_raw"], "test")
            path = root / f"test_seed{seed}.jsonl"
            path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
            raw_hashes[path.name] = sha(path); test_seed[str(seed)] = aggregate(records)
        test = summarize(test_seed); test_scored = True
    peak_allocated = torch.cuda.max_memory_allocated() / 2**30
    peak_reserved = torch.cuda.max_memory_reserved() / 2**30
    del model; torch.cuda.empty_cache()
    transform_path = root / "transforms.pt"
    torch.save({"protocol": "zsre_contextual_readout_v1", "transforms": transforms,
                "selected": selected, "calibration_hashes": calibration_hashes}, transform_path)
    training = {"seconds": time.perf_counter() - started,
                "gpu_peak_allocated_gib": peak_allocated,
                "gpu_peak_reserved_gib": peak_reserved,
                "calibration_grid": grid, "selected_config": selected,
                "calibration_embedding_hashes": calibration_hashes,
                "dev_embedding_hashes": dev_hashes, "test_embedding_hashes": test_hashes,
                "raw_rows_sha256": raw_hashes, "transforms_sha256": sha(transform_path)}
    manifest = {"protocol": "zsre_contextual_readout_v1",
                "data_manifest_sha256": sha(Path(args.data_root) / "manifest.json"),
                "model": args.model, "tokenizer": args.tokenizer,
                "loading_info": model_meta["loading_info"], "seeds": list(SEEDS),
                "regularizers": list(REGULARIZERS), "unknown_grid": list(UNKNOWN_GRID),
                "fit_split": "calibration", "gate_split": "dev",
                "test_release_rule": "score once only after all natural-readability dev checks",
                "test_scored": test_scored}
    result = {"protocol": "zsre_contextual_readout_v1", "selected_config": selected,
              "dev": dev, "dev_gate": dev_gate, "test": test, "test_scored": test_scored,
              "stage_passed": dev_gate["passed"], "training_seconds": training["seconds"],
              "terminal_status": "completed_zsre_contextual_readout"}
    write_json(root / "training.json", training); write_json(root / "training_manifest.json", manifest)
    write_json(root / "summary.json", result); return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True); parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args(); print(json.dumps(run(args), indent=2))


if __name__ == "__main__": main()
