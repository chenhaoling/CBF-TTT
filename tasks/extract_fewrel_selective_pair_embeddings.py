"""Extract frozen Qwen cross features for FewRel train and development only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from tasks.build_fewrel_selective_forgetting import load_jsonl
from tasks.cbf_checkpoint_readout import load_plain
from tasks.extract_natural_semantic_pair_cross_embeddings import (
    MAX_TOKENS, NO_TOKEN_ID, YES_TOKEN_ID, encode_pair_batch, ordered_prompts,
)
from tasks.extract_natural_semantic_surface_embeddings import (
    model_provenance, tensor_sha256, token_ids_sha256, tokenizer_provenance,
)
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "fewrel_selective_pair_cross_embeddings_v1"
PAIR_MANIFEST_SHA256 = (
    "6f1dc9c9bacc934c5e44693b59a841220e18772ef0250eb64a5afb76aee77a1d"
)
PAIR_AUDIT_SHA256 = (
    "960e706573916846fa46194bc6323dea1756a15be0cb4dc87d0e99d32ced7a3f"
)
ALLOWED_PARTITIONS = ("train", "development")
EXPECTED_PAIRS = 66658


def development_pairs(rows):
    selected = [row for row in rows if row["partition"] in ALLOWED_PARTITIONS]
    if len(selected) != EXPECTED_PAIRS:
        raise ValueError(f"unexpected train/development pair count: {len(selected)}")
    return selected


def partition_pairs(pairs, shard, num_shards):
    if num_shards < 1 or not 0 <= shard < num_shards:
        raise ValueError("invalid FewRel embedding shard")
    return [(index, row) for index, row in enumerate(pairs)
            if index % num_shards == shard]


def verify_release(pair_root):
    pair_root = Path(pair_root)
    manifest_path, audit_path = pair_root / "data/manifest.json", pair_root / "audit.json"
    manifest, audit = json.loads(manifest_path.read_text()), json.loads(audit_path.read_text())
    if any((
        sha(manifest_path) != PAIR_MANIFEST_SHA256,
        sha(audit_path) != PAIR_AUDIT_SHA256,
        audit.get("audit_passed") is not True,
        audit.get("internal_test_scored") is not False,
        audit.get("external_test_scored") is not False,
        (pair_root / "runner_exit_code.txt").read_text().strip() != "0",
        (pair_root / "terminal_status.txt").read_text().strip()
            != "completed_fewrel_selective_forgetting_pairs",
    )):
        raise ValueError("FewRel embeddings require the audited frozen pair release")
    return manifest, audit


def extract(args):
    import torch
    import transformers
    from transformers import AutoTokenizer

    pair_root, output_root = Path(args.pair_root), Path(args.output_root)
    manifest, _ = verify_release(pair_root)
    pairs_path = pair_root / "data/pairs.jsonl"
    pairs = development_pairs(load_jsonl(pairs_path))
    selected = partition_pairs(pairs, args.shard, args.num_shards)
    output_root.mkdir(parents=True, exist_ok=True)
    tensor_path = output_root / f"shard_{args.shard}.pt"
    rows_path = output_root / f"shard_{args.shard}.jsonl"
    metadata_path = output_root / f"shard_{args.shard}.audit.json"
    if any(path.exists() for path in (tensor_path, rows_path, metadata_path)):
        raise FileExistsError("FewRel embedding shard already exists")

    torch.manual_seed(2331)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if (tokenizer.encode(" yes", add_special_tokens=False) != [YES_TOKEN_ID]
            or tokenizer.encode(" no", add_special_tokens=False) != [NO_TOKEN_ID]):
        raise ValueError("registered yes/no tokens changed")
    model, raw_meta = load_plain(args.model)
    model_meta = model_provenance(raw_meta)
    if model_meta["ttt_mode"] is not False:
        raise ValueError("TTT must be disabled during FewRel feature extraction")

    vectors, margins, records, profiles = [], [], [], []
    for start in range(0, len(selected), args.batch_size):
        batch = selected[start:start + args.batch_size]
        before = time.perf_counter()
        vector, margin, prompt_ids, prompt_lengths = encode_pair_batch(
            model, tokenizer, [row for _, row in batch])
        profiles.append({
            "start": start, "size": len(batch),
            "seconds": time.perf_counter() - before,
            "max_tokens": max(max(value) for value in prompt_lengths),
        })
        for offset, ((global_index, pair), ids, lengths) in enumerate(
                zip(batch, prompt_ids, prompt_lengths)):
            value = vector[offset].contiguous()
            score = float(margin[offset])
            prompts = ordered_prompts(pair)
            vectors.append(value); margins.append(score)
            records.append({
                "pair_id": pair["pair_id"], "pair_index": global_index,
                "partition": pair["partition"],
                "prompt_sha256": [hashlib.sha256(text.encode()).hexdigest()
                                  for text in prompts],
                "prompt_token_lengths": list(lengths),
                "prompt_token_ids_sha256": [token_ids_sha256(item) for item in ids],
                "vector_sha256": tensor_sha256(value),
                "zero_shot_yes_minus_no_margin": score,
                "batch_index": len(profiles) - 1,
            })
        print(json.dumps({"shard": args.shard, "completed": len(records),
                          "total": len(selected)}), flush=True)
    matrix = torch.stack(vectors).float().contiguous()
    margin_tensor = torch.tensor(margins, dtype=torch.float32)
    if (not torch.isfinite(matrix).all() or not torch.isfinite(margin_tensor).all()
            or float((matrix.norm(dim=1) - 1).abs().max()) > 1e-5):
        raise ValueError("invalid FewRel cross embeddings")
    bundle = {
        "protocol": PROTOCOL, "shard": args.shard, "num_shards": args.num_shards,
        "pair_ids": [row["pair_id"] for row in records],
        "pair_indices": [row["pair_index"] for row in records],
        "partitions": [row["partition"] for row in records],
        "embeddings": matrix, "zero_shot_margins": margin_tensor,
    }
    temporary = tensor_path.with_suffix(".pt.tmp")
    torch.save(bundle, temporary); temporary.replace(tensor_path)
    with rows_path.open("x") as sink:
        for row in records:
            sink.write(json.dumps(row) + "\n")
    metadata = {
        "protocol": PROTOCOL, "shard": args.shard, "num_shards": args.num_shards,
        "rows": len(records), "embedding_dim": matrix.shape[1], "dtype": str(matrix.dtype),
        "allowed_partitions": list(ALLOWED_PARTITIONS), "max_tokens": MAX_TOKENS,
        "batch_size": args.batch_size, "yes_token_id": YES_TOKEN_ID,
        "no_token_id": NO_TOKEN_ID, "model": model_meta,
        "tokenizer": tokenizer_provenance(args.tokenizer, tokenizer),
        "dependencies": {
            "pair_manifest_sha256": PAIR_MANIFEST_SHA256,
            "pair_audit_sha256": PAIR_AUDIT_SHA256,
            "pairs_sha256": manifest["files_sha256"]["pairs.jsonl"],
        },
        "files_sha256": {tensor_path.name: sha(tensor_path), rows_path.name: sha(rows_path)},
        "matrix_sha256": tensor_sha256(matrix),
        "zero_shot_margins_sha256": tensor_sha256(margin_tensor),
        "batch_profiles": profiles, "seconds": time.perf_counter() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "transformers": transformers.__version__,
        "backbone_trainable": any(value.requires_grad for value in model.parameters()),
        "labels_loaded": False, "internal_test_loaded": False,
        "external_test_loaded": False, "sealed_benchmarks_loaded": False,
        "terminal_status": "completed_fewrel_pair_cross_embedding_shard",
    }
    write_json(metadata_path, metadata)
    return metadata


def load_embeddings(root):
    import torch
    rows = []
    for shard in range(2):
        bundle = torch.load(Path(root) / "embeddings" / f"shard_{shard}.pt",
                            map_location="cpu", weights_only=True)
        rows.extend((index, pair_id, partition, bundle["embeddings"][position].float(),
                     float(bundle["zero_shot_margins"][position]))
                    for position, (index, pair_id, partition) in enumerate(zip(
                        bundle["pair_indices"], bundle["pair_ids"], bundle["partitions"])))
    rows.sort(key=lambda value: value[0])
    if [row[0] for row in rows] != list(range(len(rows))):
        raise ValueError("FewRel embedding indices incomplete")
    return ([row[1] for row in rows], [row[2] for row in rows],
            torch.stack([row[3] for row in rows]), torch.tensor([row[4] for row in rows]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    result = extract(parser.parse_args())
    print(json.dumps({"status": result["terminal_status"], "shard": result["shard"],
                      "rows": result["rows"], "seconds": result["seconds"]}))


if __name__ == "__main__":
    main()
