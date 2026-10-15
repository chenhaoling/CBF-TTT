"""Extract frozen Qwen embeddings for S4 candidate and control windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from tasks.build_pair_localized_relation_memory_s4 import (
    PROTOCOL as DATA_PROTOCOL,
    load_jsonl,
)
from tasks.build_retrieval_evidence_forgetting_s3 import source_hash
from tasks.cbf_checkpoint_readout import load_plain
from tasks.pack_cbf_event_warmup import sha


PROTOCOL = "qwen_pair_localized_relation_memory_s4_embeddings_v1"
AUDIT_PROTOCOL = "qwen_pair_localized_relation_memory_s4_data_audit_v1"
DATASETS = (
    "train_candidates", "calibration_candidates",
    "train_controls", "calibration_controls",
)


def partition(rows: list[dict], shard: int, num_shards: int) -> list[tuple[int, dict]]:
    if num_shards < 1 or not 0 <= shard < num_shards:
        raise ValueError("invalid S4 embedding shard")
    return [(index, row) for index, row in enumerate(rows) if index % num_shards == shard]


def candidate_rows(mentions: list[dict]) -> list[dict]:
    output = []
    for mention in mentions:
        for candidate in mention["candidates"]:
            features = candidate["features"]
            output.append({
                "view_id": candidate["candidate_id"],
                "source_id": mention["source_row_sha256"],
                "text": candidate["model_text"],
                "relation": mention["relation"],
                "source": mention["source"],
                "dialogue_id": mention["dialogue_sha256"],
                "view": "candidate",
                "center_turn": candidate["center_turn"],
                "features": [
                    float(features["arg1_present"]), float(features["arg2_present"]),
                    features["distance_to_arg1"], features["distance_to_arg2"],
                    features["normalized_position"],
                ],
            })
    return output


def control_rows(mentions: list[dict]) -> list[dict]:
    output = []
    for mention in mentions:
        for radius in (0, 1, 2):
            view = f"deterministic_r{radius}"
            window = mention["deterministic_windows"][str(radius)]
            output.append({
                "view_id": source_hash("s4-control", mention["source_row_sha256"], view),
                "source_id": mention["source_row_sha256"],
                "text": window["model_text"],
                "relation": mention["relation"],
                "source": mention["source"],
                "dialogue_id": mention["dialogue_sha256"],
                "view": view,
                "center_turn": -1,
                "features": [0.0] * 5,
            })
        oracle = mention["oracle_window_audit"]
        if oracle["available"]:
            output.append({
                "view_id": source_hash("s4-control", mention["source_row_sha256"], "oracle"),
                "source_id": mention["source_row_sha256"],
                "text": oracle["model_text"],
                "relation": mention["relation"],
                "source": mention["source"],
                "dialogue_id": mention["dialogue_sha256"],
                "view": "oracle",
                "center_turn": -1,
                "features": [0.0] * 5,
            })
    return output


def load_rows(data_root: str | Path, dataset: str) -> tuple[list[dict], dict]:
    if dataset not in DATASETS:
        raise ValueError("S4 extractor may not load development or test")
    root = Path(data_root)
    manifest_path, audit_path = root / "manifest.json", root / "data_audit.json"
    manifest, audit = map(lambda path: json.loads(path.read_text()),
                          (manifest_path, audit_path))
    if (manifest.get("protocol") != DATA_PROTOCOL
            or audit.get("protocol") != AUDIT_PROTOCOL
            or audit.get("passed") is not True
            or audit.get("dialogre_dev_rows_parsed") != 0
            or audit.get("dialogre_test_rows_parsed") != 0):
        raise ValueError("S4 embeddings require passed sealed data audit")
    split, kind = dataset.split("_", 1)
    filename = f"{split}_localized_mentions.jsonl"
    path = root / filename
    if sha(path) != manifest["files_sha256"][filename]:
        raise ValueError("S4 embedding source file changed")
    mentions = load_jsonl(path)
    rows = candidate_rows(mentions) if kind == "candidates" else control_rows(mentions)
    if len({row["view_id"] for row in rows}) != len(rows):
        raise ValueError("S4 embedding view IDs are not unique")
    return rows, {
        "manifest_sha256": sha(manifest_path),
        "audit_sha256": sha(audit_path),
        "source_file": filename,
        "source_sha256": sha(path),
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
    vectors, indices, view_ids, source_ids, relations, sources, dialogues = (
        [], [], [], [], [], [], []
    )
    views, centers, side_features, lengths = [], [], [], []
    started = time.perf_counter()
    for start in range(0, len(selected), args.batch_size):
        batch = selected[start:start + args.batch_size]
        encoded = tokenizer(
            [row["text"] for _, row in batch], return_tensors="pt", padding=True,
            truncation=False, add_special_tokens=True,
        ).to("cuda")
        width = int(encoded.input_ids.shape[1])
        if width > args.max_length:
            raise ValueError(f"S4 window exceeds maximum length: {width}")
        with torch.inference_mode():
            hidden = model.model(**encoded, use_cache=False).last_hidden_state.float()
            mask = encoded.attention_mask.unsqueeze(-1)
            pooled = torch.nn.functional.normalize(
                (hidden * mask).sum(1) / mask.sum(1), dim=1
            )
        vectors.append(pooled.cpu().numpy().astype("float16"))
        indices.extend(index for index, _ in batch)
        view_ids.extend(row["view_id"] for _, row in batch)
        source_ids.extend(row["source_id"] for _, row in batch)
        relations.extend(row["relation"] for _, row in batch)
        sources.extend(row["source"] for _, row in batch)
        dialogues.extend(row["dialogue_id"] for _, row in batch)
        views.extend(row["view"] for _, row in batch)
        centers.extend(row["center_turn"] for _, row in batch)
        side_features.extend(row["features"] for _, row in batch)
        lengths.extend(encoded.attention_mask.sum(1).cpu().tolist())
        if start % (args.batch_size * 100) == 0:
            print(json.dumps({"dataset": args.dataset, "shard": args.shard,
                              "completed": len(view_ids), "total": len(selected)}),
                  flush=True)
    matrix = np.concatenate(vectors) if vectors else np.empty((0, 0), dtype="float16")
    np.savez_compressed(
        output,
        indices=np.asarray(indices), view_ids=np.asarray(view_ids),
        source_ids=np.asarray(source_ids), relations=np.asarray(relations),
        sources=np.asarray(sources), dialogue_ids=np.asarray(dialogues),
        views=np.asarray(views), center_turns=np.asarray(centers),
        side_features=np.asarray(side_features, dtype="float32"), embeddings=matrix,
    )
    seconds = time.perf_counter() - started
    audit = {
        "protocol": PROTOCOL,
        "dataset": args.dataset,
        "shard": args.shard,
        "num_shards": args.num_shards,
        "rows": len(view_ids),
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
        "rows_per_second": len(view_ids) / seconds,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "model_outputs_are_frozen_embeddings_only": True,
        "ttt_mode": False,
        "dialogre_dev_loaded": False,
        "dialogre_test_loaded": False,
        "pubmed_validation_loaded": False,
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "completed", "dataset": args.dataset,
                      "shard": args.shard, "rows": len(view_ids), "seconds": seconds,
                      "peak_reserved_gib": audit["peak_reserved_gib"]}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, required=True)
    extract(parser.parse_args())


if __name__ == "__main__":
    main()
