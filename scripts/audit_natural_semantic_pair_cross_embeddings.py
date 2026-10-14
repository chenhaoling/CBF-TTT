"""Audit symmetric Qwen cross-encoder pair features."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tasks.build_natural_semantic_relation_pairs import load_jsonl
from tasks.cbf_checkpoint_readout import load_plain
from tasks.evaluate_natural_semantic_embedding_pairs import (
    PAIR_MANIFEST_SHA256, PAIR_ROWS_SHA256,
)
from tasks.extract_natural_semantic_pair_cross_embeddings import (
    PROTOCOL, encode_pair_batch, ordered_prompts, partition_pairs,
)
from tasks.extract_natural_semantic_surface_embeddings import (
    model_provenance, select_audit_indices, tensor_sha256,
    token_ids_sha256, tokenizer_provenance,
)
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


def audit(args):
    import torch
    from transformers import AutoTokenizer
    pair_root, root = Path(args.pair_root), Path(args.embedding_root)
    pairs_path = pair_root / "pairs.jsonl"; pairs = load_jsonl(pairs_path)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    records, vectors, margins, audits = [], [], [], []
    structure = True
    for shard in range(2):
        tensor_path = root / "embeddings" / f"shard_{shard}.pt"
        rows_path = root / "embeddings" / f"shard_{shard}.jsonl"
        audit_path = root / "embeddings" / f"shard_{shard}.audit.json"
        bundle = torch.load(tensor_path, map_location="cpu", weights_only=True)
        rows, metadata = load_jsonl(rows_path), json.loads(audit_path.read_text())
        expected = partition_pairs(pairs, shard, 2)
        matrix = bundle["embeddings"].float(); score = bundle["zero_shot_margins"].float()
        structure &= all((
            bundle["protocol"] == PROTOCOL,
            bundle["pair_indices"] == [index for index, _ in expected],
            bundle["pair_ids"] == [row["pair_id"] for _, row in expected],
            len(rows) == len(expected) == matrix.shape[0] == score.shape[0],
            metadata["files_sha256"][tensor_path.name] == sha(tensor_path),
            metadata["files_sha256"][rows_path.name] == sha(rows_path),
            metadata["matrix_sha256"] == tensor_sha256(matrix),
            metadata["zero_shot_margins_sha256"] == tensor_sha256(score),
            metadata["backbone_trainable"] is False,
            metadata["model"]["ttt_mode"] is False,
        ))
        for position, ((global_index, pair), row) in enumerate(zip(expected, rows)):
            prompts = ordered_prompts(pair)
            ids = [tokenizer(text, add_special_tokens=False)["input_ids"] for text in prompts]
            structure &= all((
                row["pair_id"] == pair["pair_id"], row["pair_index"] == global_index,
                row["prompt_sha256"] == [hashlib.sha256(text.encode()).hexdigest()
                                         for text in prompts],
                row["prompt_token_lengths"] == [len(value) for value in ids],
                row["prompt_token_ids_sha256"] == [token_ids_sha256(value) for value in ids],
                row["vector_sha256"] == tensor_sha256(matrix[position]),
                row["zero_shot_yes_minus_no_margin"] == float(score[position]),
            ))
        records.extend(rows); vectors.append(matrix); margins.append(score); audits.append(metadata)
    ordered = sorted(range(len(records)), key=lambda index: records[index]["pair_index"])
    records = [records[index] for index in ordered]
    matrix = torch.cat(vectors)[ordered]; margins = torch.cat(margins)[ordered]
    coverage = ([row["pair_id"] for row in records] == [row["pair_id"] for row in pairs])
    valid = (bool(torch.isfinite(matrix).all()) and bool(torch.isfinite(margins).all())
             and float((matrix.norm(dim=1) - 1).abs().max()) <= 1e-5)
    provenance = all(audits[0][key] == value[key] for value in audits[1:]
                     for key in ("model", "tokenizer", "dependencies",
                                 "prompt_prefix_sha256", "yes_token_id", "no_token_id"))

    sample = select_audit_indices(len(pairs), args.audit_sample_size)
    sample_set = set(sample)
    torch.manual_seed(2231)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, raw_meta = load_plain(args.model)
    reproduced = {}
    for shard in range(2):
        selected = partition_pairs(pairs, shard, 2)
        starts = sorted({(local // args.batch_size) * args.batch_size
                         for local, (global_index, _) in enumerate(selected)
                         if global_index in sample_set})
        for start in starts:
            batch = selected[start:start + args.batch_size]
            values, scores, _, _ = encode_pair_batch(
                model, tokenizer, [row for _, row in batch])
            for offset, (global_index, _) in enumerate(batch):
                if global_index in sample_set:
                    reproduced[global_index] = (values[offset], scores[offset])
    sample_vectors = torch.stack([reproduced[index][0] for index in sample])
    sample_margins = torch.stack([reproduced[index][1] for index in sample])
    cosine = torch.nn.functional.cosine_similarity(sample_vectors, matrix[sample])
    vector_error = (sample_vectors - matrix[sample]).abs().max()
    margin_error = (sample_margins - margins[sample]).abs().max()
    checks = {
        "frozen_pair_dependencies": (
            sha(pair_root / "manifest.json") == PAIR_MANIFEST_SHA256
            and sha(pairs_path) == PAIR_ROWS_SHA256),
        "shard_structure_prompts_tokens_and_hashes": structure,
        "exact_full_pair_coverage": coverage,
        "finite_unit_vectors_and_finite_margins": valid,
        "identical_shard_provenance": provenance,
        "independent_original_batch_reproduction": (
            float(cosine.min()) >= .999999 and float(vector_error) <= 1e-6
            and float(margin_error) <= 1e-5),
        "model_and_tokenizer_provenance_reproduced": (
            model_provenance(raw_meta) == audits[0]["model"]
            and tokenizer_provenance(args.tokenizer, tokenizer) == audits[0]["tokenizer"]),
    }
    result = {
        "protocol": "audit_natural_semantic_pair_cross_embeddings_v1",
        "checks": checks, "audit_passed": all(checks.values()),
        "rows": len(records), "embedding_dim": matrix.shape[1],
        "prompt_token_lengths": {"min": min(min(row["prompt_token_lengths"])
                                              for row in records),
                                 "max": max(max(row["prompt_token_lengths"])
                                            for row in records)},
        "reproduction": {"sample_size": len(sample),
                         "minimum_cosine": float(cosine.min()),
                         "maximum_vector_absolute_error": float(vector_error),
                         "maximum_margin_absolute_error": float(margin_error)},
        "resources": {"seconds": [value["seconds"] for value in audits],
                      "peak_allocated_gib": [value["peak_allocated_gib"] for value in audits],
                      "peak_reserved_gib": [value["peak_reserved_gib"] for value in audits]},
        "model": audits[0]["model"], "tokenizer": audits[0]["tokenizer"],
        "terminal_status": "audited_natural_semantic_pair_cross_embeddings",
    }
    write_json(args.output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-root", required=True)
    parser.add_argument("--embedding-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--audit-sample-size", type=int, default=8)
    result = audit(parser.parse_args())
    print(json.dumps(result, indent=2))
    if not result["audit_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
