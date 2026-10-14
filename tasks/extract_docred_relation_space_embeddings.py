"""Extract frozen Qwen statement and training-description representations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from tasks.build_fewrel_selective_forgetting import load_jsonl
from tasks.cbf_checkpoint_readout import load_plain
from tasks.extract_natural_semantic_surface_embeddings import (
    model_provenance, tensor_sha256, token_ids_sha256, tokenizer_provenance,
)
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "docred_relation_space_last_token_embeddings_v1"
DATA_MANIFEST_SHA256 = (
    "cc6731d694a988fcf7e79af54dab89319436ccfa81181de85e5665c76a928cd0"
)
DATA_AUDIT_SHA256 = (
    "ddfc013b4dc635057c5dafeecc397aaba4a476878140bbf4aa5f76b36feabeff"
)
ALLOWED_STATEMENT_PARTITIONS = ("representation_train", "relation_development")
MAX_TOKENS = 512
STATEMENT_PREFIX = (
    "Identify the semantic relation between [SUBJECT] and [VALUE].\nEvidence: "
)
STATEMENT_SUFFIX = "\nRelation representation:"
DESCRIPTION_PREFIX = "Relation description: "
DESCRIPTION_SUFFIX = "\nRelation representation:"


def statement_prompt(surface):
    return STATEMENT_PREFIX + surface + STATEMENT_SUFFIX


def description_prompt(description):
    return DESCRIPTION_PREFIX + description + DESCRIPTION_SUFFIX


def prepare_items(instances, descriptions, manifest):
    partitions = manifest["partitions"]
    allowed_relations = set(partitions["representation_train"])
    development_relations = set(partitions["relation_development"])
    expected_descriptions = set(partitions["representation_train"])
    if {row["relation_id"] for row in descriptions} != expected_descriptions:
        raise ValueError("training relation descriptions are incomplete or contaminated")
    if len(descriptions) != len(expected_descriptions):
        raise ValueError("duplicate training relation descriptions")

    items = []
    for row in instances:
        if row["partition"] not in ALLOWED_STATEMENT_PARTITIONS:
            continue
        relation = row["relation_id"]
        expected = allowed_relations if row["partition"] == "representation_train" \
            else development_relations
        if relation not in expected:
            raise ValueError("statement relation is outside its frozen partition")
        prompt = statement_prompt(row["surface"])
        items.append({
            "item_id": row["instance_id"], "item_type": "statement",
            "partition": row["partition"], "relation_id": relation,
            "source_id": row["instance_id"], "input_sha256": hashlib.sha256(
                row["surface"].encode()).hexdigest(), "prompt": prompt,
        })
    for row in descriptions:
        prompt = description_prompt(row["description"])
        items.append({
            "item_id": f"description:{row['relation_id']}",
            "item_type": "relation_description",
            "partition": "representation_train",
            "relation_id": row["relation_id"],
            "source_id": row["relation_id"], "input_sha256": hashlib.sha256(
                row["description"].encode()).hexdigest(), "prompt": prompt,
        })
    if len({row["item_id"] for row in items}) != len(items):
        raise ValueError("duplicate DocRED embedding item IDs")
    return items


def partition_items(items, shard, num_shards):
    if num_shards < 1 or not 0 <= shard < num_shards:
        raise ValueError("invalid DocRED embedding shard")
    return [(index, row) for index, row in enumerate(items)
            if index % num_shards == shard]


def verify_release(data_root):
    data_root = Path(data_root)
    manifest_path = data_root / "data/manifest.json"
    audit_path = data_root / "audit.json"
    manifest = json.loads(manifest_path.read_text())
    audit = json.loads(audit_path.read_text())
    if any((
        sha(manifest_path) != DATA_MANIFEST_SHA256,
        sha(audit_path) != DATA_AUDIT_SHA256,
        audit.get("audit_passed") is not True,
        audit.get("relation_source_test_scored") is not False,
        audit.get("fewrel_internal_test_loaded") is not False,
        audit.get("fewrel_external_test_loaded") is not False,
        (data_root / "runner_exit_code.txt").read_text().strip() != "0",
        (data_root / "terminal_status.txt").read_text().strip()
            != "completed_docred_relation_space_data",
    )):
        raise ValueError("embeddings require the audited frozen DocRED release")
    return manifest, audit


def encode_last_token_batch(model, tokenizer, texts):
    import torch
    import torch.nn.functional as F

    encoded = tokenizer(
        texts, add_special_tokens=False, padding=True, truncation=False,
        return_attention_mask=True, return_tensors="pt")
    lengths = [int(value) for value in encoded["attention_mask"].sum(dim=1).tolist()]
    if not lengths or min(lengths) < 1 or max(lengths) > MAX_TOKENS:
        raise ValueError("invalid DocRED prompt token length")
    device = next(model.parameters()).device
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    with torch.inference_mode():
        hidden = model.model(
            input_ids=input_ids, attention_mask=attention_mask,
            use_cache=False, return_dict=True).last_hidden_state.float()
        vectors = torch.stack([hidden[row, length - 1]
                               for row, length in enumerate(lengths)])
        vectors = F.normalize(vectors, p=2, dim=-1).cpu()
    token_ids = [ids[:length].tolist() for ids, length in
                 zip(encoded["input_ids"], lengths)]
    return vectors, token_ids, lengths


def extract(args):
    import torch
    import transformers
    from transformers import AutoTokenizer

    data_root, output_root = Path(args.data_root), Path(args.output_root)
    manifest, _ = verify_release(data_root)
    instances_path = data_root / "data/instances.jsonl"
    descriptions_path = data_root / "data/train_relation_descriptions.jsonl"
    instances = load_jsonl(instances_path)
    descriptions = load_jsonl(descriptions_path)
    items = prepare_items(instances, descriptions, manifest)
    selected = partition_items(items, args.shard, args.num_shards)
    output_root.mkdir(parents=True, exist_ok=True)
    tensor_path = output_root / f"shard_{args.shard}.pt"
    rows_path = output_root / f"shard_{args.shard}.jsonl"
    metadata_path = output_root / f"shard_{args.shard}.audit.json"
    if any(path.exists() for path in (tensor_path, rows_path, metadata_path)):
        raise FileExistsError("DocRED embedding shard already exists")

    torch.manual_seed(2351)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model, raw_meta = load_plain(args.model)
    model_meta = model_provenance(raw_meta)
    if model_meta["ttt_mode"] is not False:
        raise ValueError("TTT must be disabled for frozen DocRED embeddings")

    vectors, records, profiles = [], [], []
    for start in range(0, len(selected), args.batch_size):
        batch = selected[start:start + args.batch_size]
        before = time.perf_counter()
        matrix, token_ids, lengths = encode_last_token_batch(
            model, tokenizer, [row["prompt"] for _, row in batch])
        elapsed = time.perf_counter() - before
        profiles.append({"start": start, "size": len(batch), "seconds": elapsed,
                         "max_tokens": max(lengths)})
        for offset, ((global_index, item), ids, length) in enumerate(
                zip(batch, token_ids, lengths)):
            vector = matrix[offset].contiguous()
            vectors.append(vector)
            records.append({
                "item_id": item["item_id"], "item_index": global_index,
                "item_type": item["item_type"], "partition": item["partition"],
                "relation_id": item["relation_id"], "source_id": item["source_id"],
                "input_sha256": item["input_sha256"],
                "prompt_sha256": hashlib.sha256(item["prompt"].encode()).hexdigest(),
                "token_length": length, "token_ids_sha256": token_ids_sha256(ids),
                "vector_sha256": tensor_sha256(vector),
                "batch_index": len(profiles) - 1,
            })
        print(json.dumps({"shard": args.shard, "completed": len(records),
                          "total": len(selected)}), flush=True)
    matrix = torch.stack(vectors).float().contiguous()
    if (not torch.isfinite(matrix).all()
            or float((matrix.norm(dim=1) - 1).abs().max()) > 1e-5):
        raise ValueError("invalid DocRED last-token embeddings")
    bundle = {
        "protocol": PROTOCOL, "shard": args.shard, "num_shards": args.num_shards,
        "item_ids": [row["item_id"] for row in records],
        "item_indices": [row["item_index"] for row in records],
        "item_types": [row["item_type"] for row in records],
        "partitions": [row["partition"] for row in records],
        "relation_ids": [row["relation_id"] for row in records],
        "embeddings": matrix,
    }
    temporary = tensor_path.with_suffix(".pt.tmp")
    torch.save(bundle, temporary); temporary.replace(tensor_path)
    with rows_path.open("x") as sink:
        for row in records:
            sink.write(json.dumps(row, sort_keys=True) + "\n")
    metadata = {
        "protocol": PROTOCOL, "shard": args.shard, "num_shards": args.num_shards,
        "rows": len(records), "embedding_dim": matrix.shape[1],
        "dtype": str(matrix.dtype), "max_tokens": MAX_TOKENS,
        "batch_size": args.batch_size, "allowed_statement_partitions": list(
            ALLOWED_STATEMENT_PARTITIONS),
        "item_counts": {
            "statement": sum(row["item_type"] == "statement" for row in records),
            "relation_description": sum(
                row["item_type"] == "relation_description" for row in records),
        },
        "model": model_meta, "tokenizer": tokenizer_provenance(args.tokenizer, tokenizer),
        "dependencies": {
            "data_manifest_sha256": DATA_MANIFEST_SHA256,
            "data_audit_sha256": DATA_AUDIT_SHA256,
            "instances_sha256": manifest["files_sha256"]["instances.jsonl"],
            "training_descriptions_sha256": manifest["files_sha256"][
                "train_relation_descriptions.jsonl"],
        },
        "files_sha256": {tensor_path.name: sha(tensor_path),
                         rows_path.name: sha(rows_path)},
        "matrix_sha256": tensor_sha256(matrix), "batch_profiles": profiles,
        "seconds": time.perf_counter() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "transformers": transformers.__version__,
        "backbone_trainable": any(value.requires_grad for value in model.parameters()),
        "training_relation_labels_loaded": True,
        "relation_source_test_feature_extracted": False,
        "fewrel_internal_test_loaded": False, "fewrel_external_test_loaded": False,
        "sealed_benchmarks_loaded": False,
        "terminal_status": "completed_docred_relation_space_embedding_shard",
    }
    write_json(metadata_path, metadata)
    return metadata


def load_embeddings(root):
    """Load and restore the globally registered item order from two shards."""
    import torch

    root = Path(root)
    values = []
    for shard in range(2):
        bundle = torch.load(root / "embeddings" / f"shard_{shard}.pt",
                            map_location="cpu", weights_only=True)
        values.extend((
            int(index), item_id, item_type, partition, relation_id,
            bundle["embeddings"][position].float(),
        ) for position, (index, item_id, item_type, partition, relation_id)
                      in enumerate(zip(
                          bundle["item_indices"], bundle["item_ids"],
                          bundle["item_types"], bundle["partitions"],
                          bundle["relation_ids"])))
    values.sort(key=lambda row: row[0])
    if [row[0] for row in values] != list(range(len(values))):
        raise ValueError("DocRED embedding indices are incomplete")
    records = [{"item_id": row[1], "item_type": row[2],
                "partition": row[3], "relation_id": row[4]}
               for row in values]
    return records, torch.stack([row[5] for row in values])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
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
