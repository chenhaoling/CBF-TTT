"""Extract frozen Qwen3-4B embeddings for the audited S3 support corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from tasks.cbf_checkpoint_readout import load_plain
from tasks.pack_cbf_event_warmup import sha


PROTOCOL = "qwen_retrieval_evidence_forgetting_s3_embeddings_v1"
DATA_PROTOCOL = "qwen_retrieval_evidence_forgetting_s3"
AUDIT_PROTOCOL = "qwen_retrieval_evidence_forgetting_s3_data_audit_v1"
DATASETS = ("train", "calibration", "pubmed_full", "pubmed_dropout")


def partition(rows: list[dict], shard: int, num_shards: int) -> list[tuple[int, dict]]:
    if num_shards < 1 or not 0 <= shard < num_shards:
        raise ValueError("invalid S3 embedding shard")
    return [(index, row) for index, row in enumerate(rows) if index % num_shards == shard]


def load_rows(data_root: str | Path, dataset: str) -> tuple[list[dict], dict]:
    if dataset not in DATASETS:
        raise ValueError("S3 extractor may not load development or test")
    root = Path(data_root)
    manifest_path, audit_path = root / "manifest.json", root / "data_audit.json"
    manifest = json.loads(manifest_path.read_text())
    audit = json.loads(audit_path.read_text())
    if (manifest.get("protocol") != DATA_PROTOCOL
            or audit.get("protocol") != AUDIT_PROTOCOL
            or audit.get("passed") is not True
            or audit.get("dialogre_dev_rows_parsed") != 0
            or audit.get("dialogre_test_rows_parsed") != 0
            or audit.get("pubmed_validation_rows_parsed") != 0):
        raise ValueError("S3 embeddings require passed sealed data audit")
    if dataset in {"train", "calibration"}:
        filename = f"{dataset}_mentions.jsonl"
        rows = [json.loads(line) for line in (root / filename).read_text().splitlines() if line]
        output = [{
            "source_id": row["source_row_sha256"],
            "text": row["masked_text"],
            "relation": row["relation"],
            "source": row["source"],
            "dialogue_id": row["dialogue_sha256"],
            "type_pair": row["type_pair"],
        } for row in rows]
    else:
        filename = "pubmed_unsupervised_views.jsonl"
        field = "view_full" if dataset == "pubmed_full" else "view_dropout"
        rows = [json.loads(line) for line in (root / filename).read_text().splitlines() if line]
        output = [{
            "source_id": row["row_id"],
            "text": row[field],
            "relation": "",
            "source": "pubmed_unsupervised",
            "dialogue_id": row["row_id"],
            "type_pair": "",
        } for row in rows]
    if sha(root / filename) != manifest["files_sha256"][filename]:
        raise ValueError("S3 embedding source file changed")
    if len({row["source_id"] for row in output}) != len(output):
        raise ValueError("S3 embedding source IDs are not unique")
    return output, {
        "manifest_sha256": sha(manifest_path),
        "audit_sha256": sha(audit_path),
        "source_file": filename,
        "source_sha256": sha(root / filename),
    }


def extract(args) -> None:
    import numpy as np
    import torch
    import transformers
    from transformers import AutoTokenizer

    rows, data_meta = load_rows(args.data_root, args.dataset)
    selected = partition(rows, args.shard, args.num_shards)
    output = Path(args.output)
    audit_path = output.with_suffix(".audit.json")
    if output.exists() or audit_path.exists():
        raise FileExistsError(output)
    torch.cuda.set_device(0)
    torch.manual_seed(35117)
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
    vectors, indices, ids, relations, sources, dialogues, types, lengths = (
        [], [], [], [], [], [], [], []
    )
    started = time.perf_counter()
    for start in range(0, len(selected), args.batch_size):
        batch = selected[start:start + args.batch_size]
        encoded = tokenizer(
            [row["text"] for _, row in batch],
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=True,
        ).to("cuda")
        width = int(encoded.input_ids.shape[1])
        if width > args.max_length:
            raise ValueError(f"S3 mention exceeds maximum length: {width}")
        with torch.inference_mode():
            hidden = model.model(**encoded, use_cache=False).last_hidden_state.float()
            mask = encoded.attention_mask.unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1)
            pooled = torch.nn.functional.normalize(pooled, dim=1)
        vectors.append(pooled.cpu().numpy().astype("float16"))
        indices.extend(index for index, _ in batch)
        ids.extend(row["source_id"] for _, row in batch)
        relations.extend(row["relation"] for _, row in batch)
        sources.extend(row["source"] for _, row in batch)
        dialogues.extend(row["dialogue_id"] for _, row in batch)
        types.extend(row["type_pair"] for _, row in batch)
        lengths.extend(encoded.attention_mask.sum(1).cpu().tolist())
        if start % (args.batch_size * 50) == 0:
            print(json.dumps({
                "dataset": args.dataset,
                "shard": args.shard,
                "completed": len(ids),
                "total": len(selected),
            }), flush=True)
    matrix = np.concatenate(vectors) if vectors else np.empty((0, 0), dtype="float16")
    np.savez_compressed(
        output,
        indices=np.asarray(indices),
        source_ids=np.asarray(ids),
        relations=np.asarray(relations),
        sources=np.asarray(sources),
        dialogue_ids=np.asarray(dialogues),
        type_pairs=np.asarray(types),
        embeddings=matrix,
    )
    seconds = time.perf_counter() - started
    audit = {
        "protocol": PROTOCOL,
        "dataset": args.dataset,
        "shard": args.shard,
        "num_shards": args.num_shards,
        "rows": len(ids),
        "dimension": int(matrix.shape[1]) if matrix.ndim == 2 and len(matrix) else 0,
        "indices_sha256": hashlib.sha256(json.dumps(indices).encode()).hexdigest(),
        "output_sha256": sha(output),
        "base_model": base_meta,
        "data": data_meta,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "truncation": False,
        "length_min": min(lengths),
        "length_max": max(lengths),
        "seconds": seconds,
        "rows_per_second": len(ids) / seconds,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "model_outputs_are_frozen_embeddings_only": True,
        "dialogre_dev_loaded": False,
        "dialogre_test_loaded": False,
        "pubmed_validation_loaded": False,
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": "completed",
        "dataset": args.dataset,
        "shard": args.shard,
        "rows": len(ids),
        "seconds": seconds,
        "peak_reserved_gib": audit["peak_reserved_gib"],
    }), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    extract(parser.parse_args())


if __name__ == "__main__":
    main()
