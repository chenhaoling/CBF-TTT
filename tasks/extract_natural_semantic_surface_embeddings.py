"""Extract frozen Qwen representations for natural semantic surfaces."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

from tasks.build_natural_semantic_relation_pairs import load_jsonl
from tasks.cbf_checkpoint_readout import load_plain
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "natural_semantic_surface_embeddings_v1"
PAIR_MANIFEST_SHA256 = (
    "66b21a27d437f250d66f0b3daee6dc3869c89954c1f211d5093c600c06162d01"
)
SURFACES_SHA256 = (
    "6f8efa32c829d3ccf3bc33511f503639c526d386bcd648f2ab461708c1a37f79"
)
MAX_TOKENS = 256
TOKENIZER_FILES = (
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "vocab.json", "merges.txt", "chat_template.jinja",
)


def tensor_sha256(tensor):
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def token_ids_sha256(token_ids):
    value = ",".join(map(str, token_ids)).encode()
    return hashlib.sha256(value).hexdigest()


def partition_surfaces(surfaces, shard, num_shards):
    if num_shards < 1 or not 0 <= shard < num_shards:
        raise ValueError("invalid shard")
    return [(index, row) for index, row in enumerate(surfaces)
            if index % num_shards == shard]


def select_audit_indices(size, count=32):
    """Fixed coverage across the ordered corpus, independent of outcomes."""
    if size <= 0:
        return []
    count = min(size, count)
    if count == 1:
        return [0]
    return sorted({round(index * (size - 1) / (count - 1))
                   for index in range(count)})


def tokenizer_provenance(path, tokenizer):
    root = Path(path)
    files = {name: sha(root / name) for name in TOKENIZER_FILES
             if (root / name).is_file()}
    return {
        "path": str(path),
        "class": type(tokenizer).__module__ + "." + type(tokenizer).__name__,
        "vocab_size": len(tokenizer),
        "padding_side": tokenizer.padding_side,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "files_sha256": files,
    }


def model_provenance(metadata):
    fields = {
        "path": metadata["path"],
        "class": metadata["class"],
        "config_sha256": metadata["config_sha256"],
        "source_files_sha256": metadata["source_files_sha256"],
        "excluded_adapters": metadata["excluded_adapters"],
        "parameter_count": metadata["parameter_count"],
        "attention": metadata["attention"],
        "ttt_mode": metadata["config"].get("ttt_mode"),
    }
    fields["fingerprint_sha256"] = hashlib.sha256(
        json.dumps(fields, sort_keys=True).encode()).hexdigest()
    return fields


def encode_batch(model, tokenizer, texts):
    import torch
    import torch.nn.functional as F

    encoded = tokenizer(
        texts, add_special_tokens=False, padding=True, truncation=False,
        return_attention_mask=True, return_tensors="pt")
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
        weights = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1)
        pooled = F.normalize(pooled, p=2, dim=-1).cpu()
    token_ids = [ids[:length].tolist() for ids, length in
                 zip(encoded["input_ids"], lengths)]
    return pooled, token_ids, [int(value) for value in lengths]


def extract(args):
    import torch
    import transformers
    from transformers import AutoTokenizer

    pair_root, output_root = Path(args.pair_root), Path(args.output_root)
    manifest_path, surfaces_path = (pair_root / "manifest.json",
                                    pair_root / "surfaces.jsonl")
    if sha(manifest_path) != PAIR_MANIFEST_SHA256:
        raise ValueError("frozen pair manifest changed")
    if sha(surfaces_path) != SURFACES_SHA256:
        raise ValueError("frozen surfaces changed")
    surfaces = load_jsonl(surfaces_path)
    if len({row["surface_id"] for row in surfaces}) != len(surfaces):
        raise ValueError("duplicate surface IDs")
    selected = partition_surfaces(surfaces, args.shard, args.num_shards)
    output_root.mkdir(parents=True, exist_ok=True)
    tensor_path = output_root / f"shard_{args.shard}.pt"
    rows_path = output_root / f"shard_{args.shard}.jsonl"
    audit_path = output_root / f"shard_{args.shard}.audit.json"
    if any(path.exists() for path in (tensor_path, rows_path, audit_path)):
        raise FileExistsError(f"embedding shard {args.shard} already exists")

    torch.manual_seed(2201)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model, raw_model_meta = load_plain(args.model)
    model_meta = model_provenance(raw_model_meta)
    if model_meta["ttt_mode"] is not False:
        raise ValueError("TTT must be disabled for frozen representation")

    vectors, rows, batch_profiles = [], [], []
    for start in range(0, len(selected), args.batch_size):
        batch = selected[start:start + args.batch_size]
        before = time.perf_counter()
        embedding, token_ids, lengths = encode_batch(
            model, tokenizer, [row["text"] for _, row in batch])
        elapsed = time.perf_counter() - before
        batch_profiles.append({
            "start": start, "size": len(batch), "seconds": elapsed,
            "max_tokens": max(lengths),
        })
        for offset, ((global_index, surface), ids, length) in enumerate(
                zip(batch, token_ids, lengths)):
            vector = embedding[offset].contiguous()
            vectors.append(vector)
            rows.append({
                "surface_id": surface["surface_id"],
                "surface_index": global_index,
                "text": surface["text"],
                "token_ids": ids,
                "token_length": length,
                "token_ids_sha256": token_ids_sha256(ids),
                "vector_sha256": tensor_sha256(vector),
                "batch_index": len(batch_profiles) - 1,
                "batch_seconds": elapsed,
                "seconds_per_surface": elapsed / len(batch),
            })
        print(json.dumps({"shard": args.shard, "completed": len(rows),
                          "total": len(selected)}), flush=True)
    matrix = torch.stack(vectors).to(torch.float32).contiguous()
    if not torch.isfinite(matrix).all():
        raise ValueError("non-finite embeddings")
    norms = matrix.norm(dim=1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-5, rtol=0):
        raise ValueError("embeddings are not normalized")
    bundle = {
        "protocol": PROTOCOL,
        "shard": args.shard,
        "num_shards": args.num_shards,
        "surface_ids": [row["surface_id"] for row in rows],
        "surface_indices": [row["surface_index"] for row in rows],
        "token_lengths": [row["token_length"] for row in rows],
        "embeddings": matrix,
    }
    temporary = tensor_path.with_suffix(".pt.tmp")
    torch.save(bundle, temporary)
    temporary.replace(tensor_path)
    with rows_path.open("x") as sink:
        for row in rows:
            sink.write(json.dumps(row) + "\n")
    audit = {
        "protocol": PROTOCOL,
        "shard": args.shard,
        "num_shards": args.num_shards,
        "rows": len(rows),
        "embedding_dim": matrix.shape[1],
        "dtype": str(matrix.dtype),
        "max_tokens": MAX_TOKENS,
        "observed_token_lengths": {
            "min": min(bundle["token_lengths"]),
            "max": max(bundle["token_lengths"]),
            "mean": sum(bundle["token_lengths"]) / len(rows),
            "over_128": sum(value > 128 for value in bundle["token_lengths"]),
            "over_256": sum(value > 256 for value in bundle["token_lengths"]),
        },
        "model": model_meta,
        "tokenizer": tokenizer_provenance(args.tokenizer, tokenizer),
        "dependencies": {
            "pair_manifest_sha256": sha(manifest_path),
            "surfaces_sha256": sha(surfaces_path),
        },
        "files_sha256": {
            tensor_path.name: sha(tensor_path),
            rows_path.name: sha(rows_path),
        },
        "embedding_matrix_sha256": tensor_sha256(matrix),
        "batch_profiles": batch_profiles,
        "seconds": time.perf_counter() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2 ** 30,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "backbone_trainable": any(parameter.requires_grad
                                    for parameter in model.parameters()),
        "terminal_status": "completed_natural_semantic_embedding_shard",
    }
    write_json(audit_path, audit)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    result = extract(args)
    print(json.dumps({"status": result["terminal_status"],
                      "shard": result["shard"], "rows": result["rows"],
                      "seconds": result["seconds"]}))


if __name__ == "__main__":
    main()
