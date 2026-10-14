"""Extract symmetric frozen-Qwen cross-encoder features for semantic pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from tasks.build_natural_semantic_relation_pairs import load_jsonl
from tasks.cbf_checkpoint_readout import load_plain
from tasks.evaluate_natural_semantic_embedding_pairs import (
    PAIR_MANIFEST_SHA256, PAIR_ROWS_SHA256,
)
from tasks.extract_natural_semantic_surface_embeddings import (
    model_provenance, tensor_sha256, token_ids_sha256, tokenizer_provenance,
)
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "natural_semantic_pair_cross_embeddings_v1"
MAX_TOKENS = 512
YES_TOKEN_ID = 9834
NO_TOKEN_ID = 902
PROMPT_PREFIX = (
    "Decide whether the two masked statements express the same relation "
    "about [SUBJECT].\n"
)


def pair_prompt(left, right):
    return (f"{PROMPT_PREFIX}Statement A: {left}\n"
            f"Statement B: {right}\nSame relation:")


def ordered_prompts(pair):
    return (pair_prompt(pair["left_surface"], pair["right_surface"]),
            pair_prompt(pair["right_surface"], pair["left_surface"]))


def partition_pairs(pairs, shard, num_shards):
    if num_shards < 1 or not 0 <= shard < num_shards:
        raise ValueError("invalid pair shard")
    return [(index, row) for index, row in enumerate(pairs)
            if index % num_shards == shard]


def encode_pair_batch(model, tokenizer, pairs):
    import torch
    import torch.nn.functional as F
    prompt_pairs = [ordered_prompts(pair) for pair in pairs]
    prompts = [prompt for values in prompt_pairs for prompt in values]
    encoded = tokenizer(
        prompts, add_special_tokens=False, padding=True, truncation=False,
        return_attention_mask=True, return_tensors="pt")
    lengths = encoded["attention_mask"].sum(dim=1).tolist()
    if not lengths or min(lengths) < 1 or max(lengths) > MAX_TOKENS:
        raise ValueError("invalid cross-encoder prompt length")
    device = next(model.parameters()).device
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    with torch.inference_mode():
        hidden = model.model(
            input_ids=input_ids, attention_mask=attention_mask,
            use_cache=False, return_dict=True).last_hidden_state.float()
        rows = torch.arange(len(prompts), device=device)
        last = hidden[rows, attention_mask.sum(dim=1) - 1]
        normalized = F.normalize(last, p=2, dim=-1).reshape(len(pairs), 2, -1)
        representation = F.normalize(normalized.mean(dim=1), p=2, dim=-1)
        weights = model.lm_head.weight[[YES_TOKEN_ID, NO_TOKEN_ID]].float()
        logits = last @ weights.T
        margin = (logits[:, 0] - logits[:, 1]).reshape(len(pairs), 2).mean(dim=1)
    prompt_ids = [ids[:int(length)].tolist() for ids, length in
                  zip(encoded["input_ids"], lengths)]
    prompt_ids = [(prompt_ids[2 * index], prompt_ids[2 * index + 1])
                  for index in range(len(pairs))]
    prompt_lengths = [(int(lengths[2 * index]), int(lengths[2 * index + 1]))
                      for index in range(len(pairs))]
    return representation.cpu(), margin.float().cpu(), prompt_ids, prompt_lengths


def extract(args):
    import torch
    import transformers
    from transformers import AutoTokenizer
    pair_root, output_root = Path(args.pair_root), Path(args.output_root)
    manifest_path, pairs_path = pair_root / "manifest.json", pair_root / "pairs.jsonl"
    if sha(manifest_path) != PAIR_MANIFEST_SHA256 or sha(pairs_path) != PAIR_ROWS_SHA256:
        raise ValueError("frozen semantic pair release changed")
    pairs = load_jsonl(pairs_path)
    selected = partition_pairs(pairs, args.shard, args.num_shards)
    output_root.mkdir(parents=True, exist_ok=True)
    tensor_path = output_root / f"shard_{args.shard}.pt"
    rows_path = output_root / f"shard_{args.shard}.jsonl"
    audit_path = output_root / f"shard_{args.shard}.audit.json"
    if any(path.exists() for path in (tensor_path, rows_path, audit_path)):
        raise FileExistsError("cross-embedding shard already exists")
    torch.manual_seed(2231)
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
        raise ValueError("TTT must be disabled")
    vectors, margins, rows, profiles = [], [], [], []
    for start in range(0, len(selected), args.batch_size):
        batch = selected[start:start + args.batch_size]
        before = time.perf_counter()
        vector, margin, prompt_ids, prompt_lengths = encode_pair_batch(
            model, tokenizer, [row for _, row in batch])
        elapsed = time.perf_counter() - before
        profiles.append({"start": start, "size": len(batch), "seconds": elapsed,
                         "max_tokens": max(max(value) for value in prompt_lengths)})
        for offset, ((global_index, pair), ids, lengths) in enumerate(
                zip(batch, prompt_ids, prompt_lengths)):
            value = vector[offset].contiguous(); score = float(margin[offset])
            vectors.append(value); margins.append(score)
            prompts = ordered_prompts(pair)
            rows.append({
                "pair_id": pair["pair_id"], "pair_index": global_index,
                "prompt_sha256": [hashlib.sha256(text.encode()).hexdigest()
                                  for text in prompts],
                "prompt_token_lengths": list(lengths),
                "prompt_token_ids_sha256": [token_ids_sha256(item) for item in ids],
                "vector_sha256": tensor_sha256(value),
                "zero_shot_yes_minus_no_margin": score,
                "batch_index": len(profiles) - 1,
            })
        print(json.dumps({"shard": args.shard, "completed": len(rows),
                          "total": len(selected)}), flush=True)
    matrix = torch.stack(vectors).float().contiguous()
    margin_tensor = torch.tensor(margins, dtype=torch.float32)
    if (not torch.isfinite(matrix).all() or not torch.isfinite(margin_tensor).all()
            or float((matrix.norm(dim=1) - 1).abs().max()) > 1e-5):
        raise ValueError("invalid pair cross-encoder outputs")
    bundle = {
        "protocol": PROTOCOL, "shard": args.shard, "num_shards": args.num_shards,
        "pair_ids": [row["pair_id"] for row in rows],
        "pair_indices": [row["pair_index"] for row in rows],
        "embeddings": matrix, "zero_shot_margins": margin_tensor,
    }
    temporary = tensor_path.with_suffix(".pt.tmp")
    torch.save(bundle, temporary); temporary.replace(tensor_path)
    with rows_path.open("x") as sink:
        for row in rows:
            sink.write(json.dumps(row) + "\n")
    audit = {
        "protocol": PROTOCOL, "shard": args.shard,
        "num_shards": args.num_shards, "rows": len(rows),
        "embedding_dim": matrix.shape[1], "dtype": str(matrix.dtype),
        "max_tokens": MAX_TOKENS, "batch_size": args.batch_size,
        "yes_token_id": YES_TOKEN_ID, "no_token_id": NO_TOKEN_ID,
        "prompt_prefix_sha256": hashlib.sha256(PROMPT_PREFIX.encode()).hexdigest(),
        "model": model_meta, "tokenizer": tokenizer_provenance(args.tokenizer, tokenizer),
        "dependencies": {"pair_manifest_sha256": sha(manifest_path),
                         "pairs_sha256": sha(pairs_path)},
        "files_sha256": {tensor_path.name: sha(tensor_path), rows_path.name: sha(rows_path)},
        "matrix_sha256": tensor_sha256(matrix),
        "zero_shot_margins_sha256": tensor_sha256(margin_tensor),
        "batch_profiles": profiles, "seconds": time.perf_counter() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "transformers": transformers.__version__,
        "backbone_trainable": any(value.requires_grad for value in model.parameters()),
        "terminal_status": "completed_natural_semantic_pair_cross_embedding_shard",
    }
    write_json(audit_path, audit)
    return audit


def load_cross_embeddings(root):
    import torch
    root, rows = Path(root), []
    for shard in range(2):
        bundle = torch.load(root / "embeddings" / f"shard_{shard}.pt",
                            map_location="cpu", weights_only=True)
        rows.extend((index, pair_id, bundle["embeddings"][position].float(),
                     float(bundle["zero_shot_margins"][position]))
                    for position, (index, pair_id) in enumerate(
                        zip(bundle["pair_indices"], bundle["pair_ids"])))
    rows.sort(key=lambda value: value[0])
    if [row[0] for row in rows] != list(range(len(rows))):
        raise ValueError("cross embedding indices incomplete")
    return ([row[1] for row in rows], torch.stack([row[2] for row in rows]),
            torch.tensor([row[3] for row in rows]))


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
    print(json.dumps({"status": result["terminal_status"],
                      "shard": result["shard"], "rows": result["rows"],
                      "seconds": result["seconds"]}))


if __name__ == "__main__":
    main()
