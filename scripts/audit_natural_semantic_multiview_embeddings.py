"""Audit causal and subject-local semantic embedding views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tasks.build_natural_semantic_relation_pairs import load_jsonl
from tasks.cbf_checkpoint_readout import load_plain
from tasks.extract_natural_semantic_multiview_embeddings import (
    PROTOCOL, VIEWS, encode_multiview_batch,
)
from tasks.extract_natural_semantic_surface_embeddings import (
    MAX_TOKENS, PAIR_MANIFEST_SHA256, SURFACES_SHA256,
    model_provenance, partition_surfaces, select_audit_indices,
    tensor_sha256, token_ids_sha256, tokenizer_provenance,
)
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


def audit(args):
    import torch
    from transformers import AutoTokenizer
    pair_root, root = Path(args.pair_root), Path(args.embedding_root)
    surfaces_path = pair_root / "surfaces.jsonl"
    surfaces = load_jsonl(surfaces_path)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    records, matrices, audits = [], {view: [] for view in VIEWS}, []
    structure = True
    for shard in range(2):
        tensor_path = root / "embeddings" / f"shard_{shard}.pt"
        rows_path = root / "embeddings" / f"shard_{shard}.jsonl"
        audit_path = root / "embeddings" / f"shard_{shard}.audit.json"
        bundle = torch.load(tensor_path, map_location="cpu", weights_only=True)
        rows, metadata = load_jsonl(rows_path), json.loads(audit_path.read_text())
        expected = partition_surfaces(surfaces, shard, 2)
        structure &= all((
            bundle["protocol"] == PROTOCOL, tuple(bundle["views"]) == VIEWS,
            bundle["surface_indices"] == [index for index, _ in expected],
            bundle["surface_ids"] == [row["surface_id"] for _, row in expected],
            len(rows) == len(expected), metadata["files_sha256"][tensor_path.name] == sha(tensor_path),
            metadata["files_sha256"][rows_path.name] == sha(rows_path),
            metadata["backbone_trainable"] is False,
            metadata["model"]["ttt_mode"] is False,
        ))
        for position, ((global_index, source), row) in enumerate(zip(expected, rows)):
            ids = tokenizer(source["text"], add_special_tokens=False)["input_ids"]
            structure &= all((
                row["surface_id"] == source["surface_id"],
                row["surface_index"] == global_index, row["text"] == source["text"],
                row["token_ids"] == ids, row["token_length"] == len(ids),
                0 < len(ids) <= MAX_TOKENS,
                row["token_ids_sha256"] == token_ids_sha256(ids),
            ))
            for view in VIEWS:
                vector = bundle["embeddings"][view][position].float()
                structure &= row["vector_sha256"][view] == tensor_sha256(vector)
        records.extend(rows)
        for view in VIEWS:
            matrix = bundle["embeddings"][view].float()
            structure &= metadata["matrix_sha256"][view] == tensor_sha256(matrix)
            matrices[view].append(matrix)
        audits.append(metadata)
    ordered = sorted(range(len(records)), key=lambda index: records[index]["surface_index"])
    records = [records[index] for index in ordered]
    merged = {view: torch.cat(parts, dim=0)[ordered] for view, parts in matrices.items()}
    coverage = ([row["surface_id"] for row in records]
                == [row["surface_id"] for row in surfaces])
    valid_vectors = all(
        bool(torch.isfinite(matrix).all())
        and float((matrix.norm(dim=1) - 1).abs().max()) <= 1e-5
        for matrix in merged.values())
    provenance = all(audits[0][key] == value[key] for value in audits[1:]
                     for key in ("model", "tokenizer", "dependencies", "views"))

    sample = select_audit_indices(len(surfaces), args.audit_sample_size)
    torch.manual_seed(2211)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, raw_meta = load_plain(args.model)
    strict = {view: {} for view in VIEWS}
    for shard in range(2):
        selected = partition_surfaces(surfaces, shard, 2)
        starts = sorted({(local // args.batch_size) * args.batch_size
                         for local, (global_index, _) in enumerate(selected)
                         if global_index in set(sample)})
        for start in starts:
            batch = selected[start:start + args.batch_size]
            values, _, _ = encode_multiview_batch(
                model, tokenizer, [row["text"] for _, row in batch])
            for offset, (global_index, _) in enumerate(batch):
                if global_index in sample:
                    for view in VIEWS:
                        strict[view][global_index] = values[view][offset]
    reproduction = {}
    reproduced = True
    for view in VIEWS:
        values = torch.stack([strict[view][index] for index in sample])
        expected = merged[view][sample]
        cosine = torch.nn.functional.cosine_similarity(values, expected)
        error = (values - expected).abs().max()
        reproduction[view] = {"minimum_cosine": float(cosine.min()),
                              "maximum_absolute_error": float(error)}
        reproduced &= float(cosine.min()) >= .999999 and float(error) <= 1e-6
    checks = {
        "frozen_dependencies": (
            sha(pair_root / "manifest.json") == PAIR_MANIFEST_SHA256
            and sha(surfaces_path) == SURFACES_SHA256),
        "shard_structure_tokens_and_hashes": structure,
        "exact_full_surface_coverage": coverage,
        "all_views_finite_unit_normalized": valid_vectors,
        "identical_shard_provenance": provenance,
        "independent_original_batch_reproduction": reproduced,
        "model_and_tokenizer_provenance_reproduced": (
            model_provenance(raw_meta) == audits[0]["model"]
            and tokenizer_provenance(args.tokenizer, tokenizer) == audits[0]["tokenizer"]),
    }
    result = {
        "protocol": "audit_natural_semantic_multiview_embeddings_v1",
        "checks": checks, "audit_passed": all(checks.values()),
        "rows": len(records), "views": list(VIEWS),
        "embedding_dim": next(iter(merged.values())).shape[1],
        "token_lengths": {"min": min(row["token_length"] for row in records),
                          "max": max(row["token_length"] for row in records)},
        "reproduction": reproduction,
        "resources": {"seconds": [value["seconds"] for value in audits],
                      "peak_allocated_gib": [value["peak_allocated_gib"] for value in audits],
                      "peak_reserved_gib": [value["peak_reserved_gib"] for value in audits]},
        "model": audits[0]["model"], "tokenizer": audits[0]["tokenizer"],
        "terminal_status": "audited_natural_semantic_multiview_embeddings",
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
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--audit-sample-size", type=int, default=16)
    result = audit(parser.parse_args())
    print(json.dumps(result, indent=2))
    if not result["audit_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
