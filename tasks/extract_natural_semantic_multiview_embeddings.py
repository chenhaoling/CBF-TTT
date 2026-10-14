"""Extract causal and subject-local Qwen views after mean-pooling failure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from tasks.build_natural_semantic_relation_pairs import load_jsonl
from tasks.cbf_checkpoint_readout import load_plain
from tasks.extract_natural_semantic_surface_embeddings import (
    MAX_TOKENS,
    PAIR_MANIFEST_SHA256,
    SURFACES_SHA256,
    model_provenance,
    partition_surfaces,
    tensor_sha256,
    token_ids_sha256,
    tokenizer_provenance,
)
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "natural_semantic_multiview_embeddings_v1"
VIEWS = ("mean", "last_token", "last8_mean", "subject_context8_mean")
MARKER = "[SUBJECT]"
WINDOW = 8


def subject_context_indices(offsets, text, window=WINDOW):
    marker_count = text.count(MARKER)
    if marker_count == 0:
        # Two audited source rows fail to expose the original subject string to
        # the inventory masker. Preserve coverage with a causal local fallback.
        return list(range(max(0, len(offsets) - window), len(offsets)))
    if marker_count != 1:
        raise ValueError("surface contains duplicate subject markers")
    marker_start = text.index(MARKER)
    marker_end = marker_start + len(MARKER)
    marker_tokens = [index for index, (start, end) in enumerate(offsets)
                     if end > start and not (end <= marker_start or start >= marker_end)]
    if not marker_tokens:
        raise ValueError("tokenizer offsets do not cover subject marker")
    left = list(range(max(0, marker_tokens[0] - window), marker_tokens[0]))
    right = list(range(marker_tokens[-1] + 1,
                       min(len(offsets), marker_tokens[-1] + 1 + window)))
    values = left + right
    if not values:
        raise ValueError("subject marker has no contextual token")
    return values


def encode_multiview_batch(model, tokenizer, texts):
    import torch
    import torch.nn.functional as F
    encoded = tokenizer(
        texts, add_special_tokens=False, padding=True, truncation=False,
        return_attention_mask=True, return_offsets_mapping=True,
        return_tensors="pt")
    offsets = encoded.pop("offset_mapping")
    lengths = encoded["attention_mask"].sum(dim=1).tolist()
    if not lengths or min(lengths) < 1 or max(lengths) > MAX_TOKENS:
        raise ValueError("invalid surface token length")
    device = next(model.parameters()).device
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    with torch.inference_mode():
        hidden = model.model(
            input_ids=input_ids, attention_mask=attention_mask,
            use_cache=False, return_dict=True).last_hidden_state.float()
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        mean = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        last, last8, context = [], [], []
        for row, (text, length) in enumerate(zip(texts, lengths)):
            length = int(length)
            last.append(hidden[row, length - 1])
            last8.append(hidden[row, max(0, length - 8):length].mean(dim=0))
            usable_offsets = offsets[row, :length].tolist()
            indices = subject_context_indices(usable_offsets, text)
            context.append(hidden[row, indices].mean(dim=0))
        views = {
            "mean": mean,
            "last_token": torch.stack(last),
            "last8_mean": torch.stack(last8),
            "subject_context8_mean": torch.stack(context),
        }
        views = {key: F.normalize(value, p=2, dim=-1).cpu()
                 for key, value in views.items()}
    token_ids = [ids[:int(length)].tolist() for ids, length in
                 zip(encoded["input_ids"], lengths)]
    return views, token_ids, [int(value) for value in lengths]


def extract(args):
    import torch
    import transformers
    from transformers import AutoTokenizer
    pair_root, output_root = Path(args.pair_root), Path(args.output_root)
    manifest_path, surfaces_path = (pair_root / "manifest.json",
                                    pair_root / "surfaces.jsonl")
    if sha(manifest_path) != PAIR_MANIFEST_SHA256 or sha(surfaces_path) != SURFACES_SHA256:
        raise ValueError("frozen semantic surface release changed")
    surfaces = load_jsonl(surfaces_path)
    selected = partition_surfaces(surfaces, args.shard, args.num_shards)
    output_root.mkdir(parents=True, exist_ok=True)
    tensor_path = output_root / f"shard_{args.shard}.pt"
    rows_path = output_root / f"shard_{args.shard}.jsonl"
    audit_path = output_root / f"shard_{args.shard}.audit.json"
    if any(path.exists() for path in (tensor_path, rows_path, audit_path)):
        raise FileExistsError("multiview shard already exists")
    torch.manual_seed(2211)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("offset-based multiview extraction requires a fast tokenizer")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model, raw_model_meta = load_plain(args.model)
    model_meta = model_provenance(raw_model_meta)
    if model_meta["ttt_mode"] is not False:
        raise ValueError("TTT must be disabled")

    view_rows = {view: [] for view in VIEWS}
    rows, profiles = [], []
    for start in range(0, len(selected), args.batch_size):
        batch = selected[start:start + args.batch_size]
        before = time.perf_counter()
        views, token_ids, lengths = encode_multiview_batch(
            model, tokenizer, [row["text"] for _, row in batch])
        elapsed = time.perf_counter() - before
        profiles.append({"start": start, "size": len(batch), "seconds": elapsed,
                         "max_tokens": max(lengths)})
        for offset, ((global_index, surface), ids, length) in enumerate(
                zip(batch, token_ids, lengths)):
            vectors = {view: views[view][offset].contiguous() for view in VIEWS}
            for view, vector in vectors.items():
                view_rows[view].append(vector)
            rows.append({
                "surface_id": surface["surface_id"], "surface_index": global_index,
                "text": surface["text"], "token_ids": ids, "token_length": length,
                "subject_marker_count": surface["text"].count(MARKER),
                "token_ids_sha256": token_ids_sha256(ids),
                "vector_sha256": {view: tensor_sha256(vector)
                                  for view, vector in vectors.items()},
                "batch_index": len(profiles) - 1,
            })
        print(json.dumps({"shard": args.shard, "completed": len(rows),
                          "total": len(selected)}), flush=True)
    matrices = {view: torch.stack(values).float().contiguous()
                for view, values in view_rows.items()}
    for view, matrix in matrices.items():
        if not torch.isfinite(matrix).all() or float((matrix.norm(dim=1) - 1).abs().max()) > 1e-5:
            raise ValueError(f"invalid {view} vectors")
    bundle = {
        "protocol": PROTOCOL, "views": VIEWS, "shard": args.shard,
        "num_shards": args.num_shards,
        "surface_ids": [row["surface_id"] for row in rows],
        "surface_indices": [row["surface_index"] for row in rows],
        "token_lengths": [row["token_length"] for row in rows],
        "embeddings": matrices,
    }
    temporary = tensor_path.with_suffix(".pt.tmp")
    torch.save(bundle, temporary); temporary.replace(tensor_path)
    with rows_path.open("x") as sink:
        for row in rows:
            sink.write(json.dumps(row) + "\n")
    audit = {
        "protocol": PROTOCOL, "views": list(VIEWS), "shard": args.shard,
        "num_shards": args.num_shards, "rows": len(rows),
        "embedding_dim": next(iter(matrices.values())).shape[1],
        "dtype": str(next(iter(matrices.values())).dtype),
        "max_tokens": MAX_TOKENS, "subject_window": WINDOW,
        "subject_marker_counts": {
            "present_once": sum(row["subject_marker_count"] == 1 for row in rows),
            "missing_fallback_last8": sum(row["subject_marker_count"] == 0 for row in rows),
            "duplicate": sum(row["subject_marker_count"] > 1 for row in rows),
        },
        "model": model_meta, "tokenizer": tokenizer_provenance(args.tokenizer, tokenizer),
        "dependencies": {"pair_manifest_sha256": sha(manifest_path),
                         "surfaces_sha256": sha(surfaces_path)},
        "files_sha256": {tensor_path.name: sha(tensor_path),
                         rows_path.name: sha(rows_path)},
        "matrix_sha256": {view: tensor_sha256(matrix)
                          for view, matrix in matrices.items()},
        "batch_size": args.batch_size, "batch_profiles": profiles,
        "seconds": time.perf_counter() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "transformers": transformers.__version__,
        "backbone_trainable": any(value.requires_grad for value in model.parameters()),
        "terminal_status": "completed_natural_semantic_multiview_embedding_shard",
    }
    write_json(audit_path, audit)
    return audit


def load_multiview_embeddings(root):
    import torch
    root, rows = Path(root), []
    for shard in range(2):
        bundle = torch.load(root / "embeddings" / f"shard_{shard}.pt",
                            map_location="cpu", weights_only=True)
        rows.extend((index, surface_id,
                     {view: bundle["embeddings"][view][position].float()
                      for view in VIEWS})
                    for position, (index, surface_id) in enumerate(
                        zip(bundle["surface_indices"], bundle["surface_ids"])))
    rows.sort(key=lambda value: value[0])
    if [row[0] for row in rows] != list(range(len(rows))):
        raise ValueError("multiview embedding indices incomplete")
    return ([row[1] for row in rows],
            {view: torch.stack([row[2][view] for row in rows]) for view in VIEWS})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    result = extract(parser.parse_args())
    print(json.dumps({"status": result["terminal_status"],
                      "shard": result["shard"], "rows": result["rows"],
                      "seconds": result["seconds"]}))


if __name__ == "__main__":
    main()
