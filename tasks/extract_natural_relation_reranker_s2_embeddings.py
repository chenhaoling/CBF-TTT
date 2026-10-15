"""Extract frozen-Qwen masked-mention embeddings for S2 retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from tasks.cbf_checkpoint_readout import load_plain
from tasks.pack_cbf_event_warmup import sha
from tasks.train_natural_relation_reranker_s2 import DATA_AUDIT_PROTOCOL, DATA_PROTOCOL


PROTOCOL = "qwen_natural_relation_reranker_s2_embeddings_v1"
SELECTION_PROTOCOL = "qwen_natural_relation_reranker_s2_selection_v1"


def partition(rows: list[dict], shard: int, num_shards: int):
    if num_shards < 1 or not 0 <= shard < num_shards:
        raise ValueError("invalid S2 embedding shard")
    return [(index, row) for index, row in enumerate(rows) if index % num_shards == shard]


def unique_mentions(rows: list[dict]) -> list[dict]:
    values = {}
    for row in rows:
        for side in ("left", "right"):
            source_id = row[f"{side}_source_row_sha256"]
            value = {"source_id": source_id, "text": row[f"{side}_text"],
                     "relation": row[f"{side}_relation_audit"]}
            if source_id in values and values[source_id] != value:
                raise ValueError("inconsistent S2 source mention")
            values[source_id] = value
    return [values[key] for key in sorted(values)]


def load_mentions(data_root: str | Path, split: str, selection: str | None) -> tuple[list[dict], dict]:
    if split not in ("calibration", "development"):
        raise ValueError("S2 embedding extractor may not load test")
    root = Path(data_root)
    manifest = json.loads((root / "manifest.json").read_text())
    audit = json.loads((root / "data_audit.json").read_text())
    if (manifest.get("protocol") != DATA_PROTOCOL
            or audit.get("protocol") != DATA_AUDIT_PROTOCOL or audit.get("passed") is not True
            or audit.get("semeval_test_rows_parsed") != 0):
        raise ValueError("S2 embeddings require audited sealed data")
    path = root / f"{split}_relation_pairs.jsonl"
    if sha(path) != manifest["files_sha256"][path.name]:
        raise ValueError("S2 embedding source changed")
    selection_hash = None
    if split == "development":
        if not selection:
            raise ValueError("development embeddings require frozen selection")
        selected = json.loads(Path(selection).read_text())
        if (selected.get("protocol") != SELECTION_PROTOCOL
                or selected.get("development_loaded") is not False
                or selected.get("semeval_test_loaded") is not False
                or selected["data_manifest_sha256"] != sha(root / "manifest.json")):
            raise ValueError("invalid S2 selection for development embeddings")
        selection_hash = sha(selection)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    return unique_mentions(rows), {
        "manifest_sha256": sha(root / "manifest.json"),
        "audit_sha256": sha(root / "data_audit.json"), "split_sha256": sha(path),
        "selection_sha256": selection_hash,
    }


def extract(args):
    import numpy as np
    import torch
    import transformers
    from transformers import AutoTokenizer

    mentions, data_meta = load_mentions(args.data_root, args.split, args.selection)
    selected = partition(mentions, args.shard, args.num_shards)
    output = Path(args.output)
    audit_path = output.with_suffix(".audit.json")
    if output.exists() or audit_path.exists():
        raise FileExistsError(output)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model, base_meta = load_plain(args.model)
    model.config.use_cache = False
    model = model.eval()
    vectors, ids, relations, indices, lengths = [], [], [], [], []
    started = time.perf_counter()
    for start in range(0, len(selected), args.batch_size):
        batch = selected[start:start + args.batch_size]
        encoded = tokenizer([row["text"] for _, row in batch], return_tensors="pt",
                            padding=True, truncation=False, add_special_tokens=True).to("cuda")
        if encoded.input_ids.shape[1] > args.max_length:
            raise ValueError("S2 mention exceeds embedding maximum length")
        with torch.inference_mode():
            hidden = model.model(**encoded, use_cache=False).last_hidden_state.float()
            mask = encoded.attention_mask.unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1)
            pooled = torch.nn.functional.normalize(pooled, dim=1)
        vectors.append(pooled.cpu().numpy().astype("float16"))
        indices.extend(index for index, _ in batch)
        ids.extend(row["source_id"] for _, row in batch)
        relations.extend(row["relation"] for _, row in batch)
        lengths.extend(encoded.attention_mask.sum(1).cpu().tolist())
        if start % (args.batch_size * 50) == 0:
            print(json.dumps({"shard": args.shard, "completed": len(ids),
                              "total": len(selected)}), flush=True)
    matrix = np.concatenate(vectors) if vectors else np.empty((0, 0), dtype="float16")
    np.savez_compressed(output, indices=np.asarray(indices), source_ids=np.asarray(ids),
                        relations=np.asarray(relations), embeddings=matrix)
    audit = {
        "protocol": PROTOCOL, "split": args.split, "shard": args.shard,
        "num_shards": args.num_shards, "rows": len(ids),
        "dimension": int(matrix.shape[1]) if matrix.ndim == 2 and len(matrix) else 0,
        "indices_sha256": hashlib.sha256(json.dumps(indices).encode()).hexdigest(),
        "output_sha256": sha(output), "base_model": base_meta, "data": data_meta,
        "length_min": min(lengths), "length_max": max(lengths),
        "seconds": time.perf_counter() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "transformers": transformers.__version__, "development_model_outputs_loaded":
            args.split == "development", "semeval_test_loaded": False,
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "completed", "split": args.split,
                      "shard": args.shard, "rows": len(ids),
                      "seconds": audit["seconds"]}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("calibration", "development"), required=True)
    parser.add_argument("--selection")
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=384)
    extract(parser.parse_args())


if __name__ == "__main__":
    main()
