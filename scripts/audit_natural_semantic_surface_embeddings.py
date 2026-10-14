"""Independently audit frozen natural semantic surface embeddings."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from tasks.build_natural_semantic_relation_pairs import load_jsonl
from tasks.extract_natural_semantic_surface_embeddings import (
    MAX_TOKENS,
    PAIR_MANIFEST_SHA256,
    PROTOCOL,
    SURFACES_SHA256,
    encode_batch,
    model_provenance,
    partition_surfaces,
    select_audit_indices,
    tensor_sha256,
    token_ids_sha256,
    tokenizer_provenance,
)
from tasks.cbf_checkpoint_readout import load_plain
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


def audit(args):
    import torch
    from transformers import AutoTokenizer

    pair_root, embedding_root = Path(args.pair_root), Path(args.embedding_root)
    manifest_path, surfaces_path = (pair_root / "manifest.json",
                                    pair_root / "surfaces.jsonl")
    surfaces = load_jsonl(surfaces_path)
    surface_index = {row["surface_id"]: row for row in surfaces}
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    all_rows, all_vectors, shard_audits = [], [], []
    structural = True
    for shard in range(args.num_shards):
        tensor_path = embedding_root / f"shard_{shard}.pt"
        rows_path = embedding_root / f"shard_{shard}.jsonl"
        audit_path = embedding_root / f"shard_{shard}.audit.json"
        bundle = torch.load(tensor_path, map_location="cpu", weights_only=True)
        rows = load_jsonl(rows_path)
        shard_audit = json.loads(audit_path.read_text())
        expected = partition_surfaces(surfaces, shard, args.num_shards)
        vectors = bundle["embeddings"].float().contiguous()
        expected_ids = [row["surface_id"] for _, row in expected]
        structural &= all((
            bundle["protocol"] == PROTOCOL,
            bundle["shard"] == shard,
            bundle["num_shards"] == args.num_shards,
            bundle["surface_ids"] == expected_ids,
            bundle["surface_indices"] == [index for index, _ in expected],
            len(rows) == len(expected) == vectors.shape[0],
            vectors.ndim == 2 and vectors.shape[1] == shard_audit["embedding_dim"],
            vectors.dtype == torch.float32,
            shard_audit["files_sha256"][tensor_path.name] == sha(tensor_path),
            shard_audit["files_sha256"][rows_path.name] == sha(rows_path),
            shard_audit["embedding_matrix_sha256"] == tensor_sha256(vectors),
            shard_audit["terminal_status"]
                == "completed_natural_semantic_embedding_shard",
            shard_audit["backbone_trainable"] is False,
            shard_audit["model"]["ttt_mode"] is False,
        ))
        for position, row in enumerate(rows):
            source = surface_index.get(row["surface_id"])
            ids = tokenizer(source["text"], add_special_tokens=False)["input_ids"] \
                if source else []
            structural &= all((
                source is not None,
                row["surface_index"] == expected[position][0],
                row["text"] == source["text"],
                row["token_ids"] == ids,
                row["token_length"] == len(ids) == bundle["token_lengths"][position],
                0 < len(ids) <= MAX_TOKENS,
                row["token_ids_sha256"] == token_ids_sha256(ids),
                row["vector_sha256"] == tensor_sha256(vectors[position]),
                math.isfinite(row["seconds_per_surface"]),
                row["seconds_per_surface"] > 0,
            ))
        all_rows.extend(rows)
        all_vectors.append(vectors)
        shard_audits.append(shard_audit)
    ordered = sorted(zip(all_rows, torch.cat(all_vectors)),
                     key=lambda value: value[0]["surface_index"])
    merged_rows = [value[0] for value in ordered]
    merged_vectors = torch.stack([value[1] for value in ordered])
    norms = merged_vectors.norm(dim=1)
    coverage = (
        len(merged_rows) == len(surfaces)
        and [row["surface_id"] for row in merged_rows]
            == [row["surface_id"] for row in surfaces]
        and len({row["surface_id"] for row in merged_rows}) == len(surfaces)
    )
    finite_normalized = bool(torch.isfinite(merged_vectors).all()) \
        and float((norms - 1).abs().max()) <= 1e-5
    provenance_equal = all(
        shard_audits[0][key] == value[key]
        for value in shard_audits[1:]
        for key in ("model", "tokenizer", "dependencies", "embedding_dim", "dtype")
    )

    # Independently reload the frozen model and reproduce a fixed 32-surface sample.
    sample_positions = select_audit_indices(len(surfaces), args.audit_sample_size)
    torch.manual_seed(2201)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, model_meta = load_plain(args.model)
    sample_vectors, sample_ids, sample_lengths = encode_batch(
        model, tokenizer, [surfaces[index]["text"] for index in sample_positions])
    expected_sample = merged_vectors[sample_positions]
    cosine = torch.nn.functional.cosine_similarity(sample_vectors, expected_sample)
    max_abs = (sample_vectors - expected_sample).abs().max(dim=1).values
    sample_reproduced = all((
        sample_lengths == [merged_rows[index]["token_length"]
                           for index in sample_positions],
        sample_ids == [merged_rows[index]["token_ids"]
                       for index in sample_positions],
        float(cosine.min()) >= 0.99999,
        float(max_abs.max()) <= 0.002,
        model_provenance(model_meta) == shard_audits[0]["model"],
        tokenizer_provenance(args.tokenizer, tokenizer)
            == shard_audits[0]["tokenizer"],
    ))
    checks = {
        "frozen_pair_dependencies": (
            sha(manifest_path) == PAIR_MANIFEST_SHA256
            and sha(surfaces_path) == SURFACES_SHA256
            and all(value["dependencies"] == {
                "pair_manifest_sha256": PAIR_MANIFEST_SHA256,
                "surfaces_sha256": SURFACES_SHA256,
            } for value in shard_audits)),
        "shard_structure_hashes_and_tokens": structural,
        "exact_full_surface_coverage": coverage,
        "finite_unit_normalized_fp32_vectors": finite_normalized,
        "identical_model_tokenizer_and_representation_provenance": provenance_equal,
        "independent_fixed_sample_reproduction": sample_reproduced,
        "no_length_exclusion_or_truncation": (
            max(row["token_length"] for row in merged_rows) <= MAX_TOKENS
            and sum(row["token_length"] > MAX_TOKENS for row in merged_rows) == 0),
    }
    result = {
        "protocol": "audit_natural_semantic_surface_embeddings_v1",
        "checks": checks,
        "audit_passed": all(checks.values()),
        "rows": len(merged_rows),
        "embedding_dim": int(merged_vectors.shape[1]),
        "token_lengths": {
            "min": min(row["token_length"] for row in merged_rows),
            "max": max(row["token_length"] for row in merged_rows),
            "over_128": sum(row["token_length"] > 128 for row in merged_rows),
            "over_256": sum(row["token_length"] > 256 for row in merged_rows),
        },
        "norm_max_absolute_error": float((norms - 1).abs().max()),
        "reproduction": {
            "sample_size": len(sample_positions),
            "positions": sample_positions,
            "minimum_cosine": float(cosine.min()),
            "maximum_absolute_error": float(max_abs.max()),
        },
        "resources": {
            "seconds": [value["seconds"] for value in shard_audits],
            "peak_allocated_gib": [value["peak_allocated_gib"]
                                   for value in shard_audits],
            "peak_reserved_gib": [value["peak_reserved_gib"]
                                  for value in shard_audits],
            "gpu": [value["gpu"] for value in shard_audits],
        },
        "model": shard_audits[0]["model"],
        "tokenizer": shard_audits[0]["tokenizer"],
        "shard_audits_sha256": {
            f"shard_{index}.audit.json": sha(
                embedding_root / f"shard_{index}.audit.json")
            for index in range(args.num_shards)},
        "terminal_status": "audited_natural_semantic_surface_embeddings",
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
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--audit-sample-size", type=int, default=32)
    args = parser.parse_args()
    result = audit(args)
    print(json.dumps(result, indent=2))
    if not result["audit_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
