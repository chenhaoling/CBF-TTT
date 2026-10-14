"""Independently audit train/development DocRED last-token embeddings."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tasks.build_fewrel_selective_forgetting import load_jsonl
from tasks.cbf_checkpoint_readout import load_plain
from tasks.extract_docred_relation_space_embeddings import (
    ALLOWED_STATEMENT_PARTITIONS, DATA_AUDIT_SHA256, DATA_MANIFEST_SHA256,
    MAX_TOKENS, PROTOCOL, encode_last_token_batch, partition_items,
    prepare_items, verify_release,
)
from tasks.extract_natural_semantic_surface_embeddings import (
    model_provenance, select_audit_indices, tensor_sha256, token_ids_sha256,
    tokenizer_provenance,
)
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


def audit(args):
    import torch
    from transformers import AutoTokenizer

    data_root, root = Path(args.data_root), Path(args.embedding_root)
    manifest, data_audit = verify_release(data_root)
    instances_path = data_root / "data/instances.jsonl"
    descriptions_path = data_root / "data/train_relation_descriptions.jsonl"
    items = prepare_items(load_jsonl(instances_path), load_jsonl(descriptions_path),
                          manifest)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    records, matrices, metadata = [], [], []
    structure = True
    for shard in range(2):
        tensor_path = root / "embeddings" / f"shard_{shard}.pt"
        rows_path = root / "embeddings" / f"shard_{shard}.jsonl"
        metadata_path = root / "embeddings" / f"shard_{shard}.audit.json"
        bundle = torch.load(tensor_path, map_location="cpu", weights_only=True)
        rows = load_jsonl(rows_path)
        meta = json.loads(metadata_path.read_text())
        expected = partition_items(items, shard, 2)
        matrix = bundle["embeddings"].float()
        structure &= all((
            bundle["protocol"] == PROTOCOL,
            bundle["item_indices"] == [index for index, _ in expected],
            bundle["item_ids"] == [row["item_id"] for _, row in expected],
            bundle["item_types"] == [row["item_type"] for _, row in expected],
            bundle["partitions"] == [row["partition"] for _, row in expected],
            bundle["relation_ids"] == [row["relation_id"] for _, row in expected],
            len(rows) == len(expected) == matrix.shape[0],
            meta["files_sha256"][tensor_path.name] == sha(tensor_path),
            meta["files_sha256"][rows_path.name] == sha(rows_path),
            meta["matrix_sha256"] == tensor_sha256(matrix),
            meta["backbone_trainable"] is False,
            meta["relation_source_test_feature_extracted"] is False,
            meta["fewrel_internal_test_loaded"] is False,
            meta["fewrel_external_test_loaded"] is False,
        ))
        for position, ((global_index, item), row) in enumerate(zip(expected, rows)):
            ids = tokenizer(item["prompt"], add_special_tokens=False)["input_ids"]
            structure &= all((
                row["item_id"] == item["item_id"],
                row["item_index"] == global_index,
                row["item_type"] == item["item_type"],
                row["partition"] == item["partition"],
                row["relation_id"] == item["relation_id"],
                row["source_id"] == item["source_id"],
                row["input_sha256"] == item["input_sha256"],
                row["prompt_sha256"] == hashlib.sha256(
                    item["prompt"].encode()).hexdigest(),
                row["token_length"] == len(ids), 0 < len(ids) <= MAX_TOKENS,
                row["token_ids_sha256"] == token_ids_sha256(ids),
                row["vector_sha256"] == tensor_sha256(matrix[position]),
            ))
        records.extend(rows); matrices.append(matrix); metadata.append(meta)

    combined = torch.cat(matrices)
    order = sorted(range(len(records)), key=lambda index: records[index]["item_index"])
    records = [records[index] for index in order]
    matrix = combined[order]
    coverage = (
        [row["item_id"] for row in records] == [row["item_id"] for row in items]
        and len({row["item_id"] for row in records}) == len(items))
    valid = (bool(torch.isfinite(matrix).all())
             and float((matrix.norm(dim=1) - 1).abs().max()) <= 1e-5)
    provenance = all(
        metadata[0][key] == value[key] for value in metadata[1:]
        for key in ("model", "tokenizer", "dependencies",
                    "allowed_statement_partitions", "max_tokens"))

    sample = select_audit_indices(len(items), args.audit_sample_size)
    sample_set = set(sample)
    torch.manual_seed(2351)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, raw_meta = load_plain(args.model)
    reproduced = {}
    for shard in range(2):
        selected = partition_items(items, shard, 2)
        starts = sorted({(local // args.batch_size) * args.batch_size
                         for local, (global_index, _) in enumerate(selected)
                         if global_index in sample_set})
        for start in starts:
            batch = selected[start:start + args.batch_size]
            values, _, _ = encode_last_token_batch(
                model, tokenizer, [row["prompt"] for _, row in batch])
            for offset, (global_index, _) in enumerate(batch):
                if global_index in sample_set:
                    reproduced[global_index] = values[offset]
    sample_vectors = torch.stack([reproduced[index] for index in sample])
    cosine = torch.nn.functional.cosine_similarity(sample_vectors, matrix[sample])
    vector_error = (sample_vectors - matrix[sample]).abs().max()

    statement_rows = [row for row in records if row["item_type"] == "statement"]
    description_rows = [row for row in records
                        if row["item_type"] == "relation_description"]
    training_relations = set(manifest["partitions"]["representation_train"])
    development_relations = set(manifest["partitions"]["relation_development"])
    source_test_relations = set(manifest["partitions"]["relation_source_test"])
    checks = {
        "frozen_audited_data_release": (
            sha(data_root / "data/manifest.json") == DATA_MANIFEST_SHA256
            and sha(data_root / "audit.json") == DATA_AUDIT_SHA256
            and data_audit["audit_passed"] is True),
        "shard_structure_prompts_tokens_and_hashes": structure,
        "exact_item_coverage": coverage,
        "finite_unit_last_token_vectors": valid,
        "identical_shard_provenance": provenance,
        "only_train_descriptions": (
            len(description_rows) == len(training_relations)
            and {row["relation_id"] for row in description_rows} == training_relations),
        "only_train_and_development_statements": (
            {row["partition"] for row in statement_rows}
                == set(ALLOWED_STATEMENT_PARTITIONS)
            and {row["relation_id"] for row in statement_rows}.isdisjoint(
                source_test_relations)
            and {row["relation_id"] for row in statement_rows}
                == training_relations | development_relations),
        "no_test_feature_or_score": all(
            meta["relation_source_test_feature_extracted"] is False
            and meta["fewrel_internal_test_loaded"] is False
            and meta["fewrel_external_test_loaded"] is False
            and meta["sealed_benchmarks_loaded"] is False
            for meta in metadata),
        "independent_original_batch_reproduction": (
            float(cosine.min()) >= .999999 and float(vector_error) <= 1e-6),
        "model_and_tokenizer_provenance_reproduced": (
            model_provenance(raw_meta) == metadata[0]["model"]
            and tokenizer_provenance(args.tokenizer, tokenizer)
                == metadata[0]["tokenizer"]),
        "dependency_hashes_reproduced": all(
            meta["dependencies"] == {
                "data_manifest_sha256": DATA_MANIFEST_SHA256,
                "data_audit_sha256": DATA_AUDIT_SHA256,
                "instances_sha256": manifest["files_sha256"]["instances.jsonl"],
                "training_descriptions_sha256": manifest["files_sha256"][
                    "train_relation_descriptions.jsonl"],
            } for meta in metadata),
    }
    result = {
        "protocol": "audit_docred_relation_space_last_token_embeddings_v1",
        "audit_passed": all(checks.values()), "checks": checks,
        "rows": len(records), "embedding_dim": matrix.shape[1],
        "item_counts": {"statement": len(statement_rows),
                        "relation_description": len(description_rows)},
        "statement_partition_counts": {
            partition: sum(row["partition"] == partition for row in statement_rows)
            for partition in ALLOWED_STATEMENT_PARTITIONS},
        "token_lengths": {"min": min(row["token_length"] for row in records),
                          "max": max(row["token_length"] for row in records)},
        "reproduction": {"sample_size": len(sample),
                         "minimum_cosine": float(cosine.min()),
                         "maximum_vector_absolute_error": float(vector_error)},
        "dependencies": {"data_manifest_sha256": DATA_MANIFEST_SHA256,
                         "data_audit_sha256": DATA_AUDIT_SHA256},
        "resources": {f"shard_{meta['shard']}": {
            key: meta[key] for key in ("seconds", "peak_allocated_gib",
                                       "peak_reserved_gib", "gpu")}
            for meta in metadata},
        "relation_source_test_feature_extracted": False,
        "fewrel_internal_test_loaded": False, "fewrel_external_test_loaded": False,
        "sealed_benchmarks_loaded": False,
        "terminal_status": "completed_docred_relation_space_embedding_audit",
    }
    write_json(Path(args.output), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--embedding-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--audit-sample-size", type=int, default=16)
    parser.add_argument("--output", required=True)
    result = audit(parser.parse_args())
    print(json.dumps(result, indent=2))
    if not result["audit_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
