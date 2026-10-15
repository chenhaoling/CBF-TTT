"""Shard-safe cached candidate scorer for the S2 binary relation reranker."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from tasks.cbf_checkpoint_readout import load_plain
from tasks.pack_cbf_event_warmup import sha
from tasks.train_natural_relation_reranker_s2 import (
    DATA_AUDIT_PROTOCOL, DATA_PROTOCOL, canonical_answer, prompt,
)


PROTOCOL = "qwen_natural_relation_reranker_s2_scores_v1"
SELECTION_PROTOCOL = "qwen_natural_relation_reranker_s2_selection_v1"


def candidates() -> list[bool]:
    return [False, True]


def partition(rows: list[dict], shard: int, num_shards: int):
    if num_shards < 1 or not 0 <= shard < num_shards:
        raise ValueError("invalid S2 score shard")
    return [(index, row) for index, row in enumerate(rows) if index % num_shards == shard]


def load_split(data_root: str | Path, split: str, selection: str | None,
               adapter: str | None) -> tuple[list[dict], dict]:
    if split not in ("calibration", "development"):
        raise ValueError("S2 scorer may not load test")
    root = Path(data_root)
    manifest = json.loads((root / "manifest.json").read_text())
    audit = json.loads((root / "data_audit.json").read_text())
    if (manifest.get("protocol") != DATA_PROTOCOL
            or audit.get("protocol") != DATA_AUDIT_PROTOCOL or audit.get("passed") is not True
            or audit.get("semeval_test_rows_parsed") != 0
            or manifest.get("semeval_test_parsed") is not False):
        raise ValueError("S2 scorer requires audited sealed data")
    path = root / f"{split}_relation_pairs.jsonl"
    if sha(path) != manifest["files_sha256"][path.name]:
        raise ValueError("S2 score split hash mismatch")
    if split == "development":
        if not selection:
            raise ValueError("development requires frozen S2 calibration selection")
        selected = json.loads(Path(selection).read_text())
        if (selected.get("protocol") != SELECTION_PROTOCOL
                or selected.get("development_loaded") is not False
                or selected.get("semeval_test_loaded") is not False
                or selected["data_manifest_sha256"] != sha(root / "manifest.json")):
            raise ValueError("invalid S2 calibration selection")
        if adapter and str(Path(adapter).resolve()) != str(Path(selected["selected_adapter"]).resolve()):
            raise ValueError("development adapter differs from frozen S2 selection")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if len(rows) != manifest["counts"][split]["relation_pairs"]:
        raise ValueError("unexpected S2 score split length")
    return rows, {"manifest_sha256": sha(root / "manifest.json"),
                  "audit_sha256": sha(root / "data_audit.json"),
                  "split_sha256": sha(path)}


def cached_candidate_nll(model, rows, tokenizer, max_length):
    import torch
    import torch.nn.functional as functional

    choices = candidates()
    old_padding = tokenizer.padding_side
    tokenizer.padding_side = "right"
    encoded = tokenizer([prompt(row) for row in rows], return_tensors="pt", padding=True,
                        add_special_tokens=True).to("cuda")
    tokenizer.padding_side = old_padding
    targets = []
    for _ in rows:
        targets.extend(tokenizer(canonical_answer(choice) + tokenizer.eos_token,
                                 add_special_tokens=False)["input_ids"]
                       for choice in choices)
    if max(encoded.input_ids.shape[1] + len(value) for value in targets) > max_length:
        raise ValueError("S2 cached candidate exceeds registered maximum length")
    width = max(len(value) for value in targets)
    target_ids = torch.tensor([value + [tokenizer.pad_token_id] * (width - len(value))
                               for value in targets], dtype=torch.long, device="cuda")
    target_mask = torch.tensor([[1] * len(value) + [0] * (width - len(value))
                                for value in targets], dtype=torch.long, device="cuda")
    with torch.inference_mode():
        prefix = model(**encoded, use_cache=True)
        prompt_lengths = encoded.attention_mask.sum(1)
        last_indices = prompt_lengths - 1
        first_logits = prefix.logits[
            torch.arange(len(rows), device="cuda"), last_indices
        ].float().repeat_interleave(len(choices), dim=0)
        first_loss = functional.cross_entropy(first_logits, target_ids[:, 0], reduction="none")
        cache = prefix.past_key_values
        del first_logits, prefix
        cache.batch_repeat_interleave(len(choices))
        repeated_mask = encoded.attention_mask.repeat_interleave(len(choices), dim=0)
        continuation_mask = torch.cat((repeated_mask, target_mask[:, :-1]), dim=1)
        positions = (prompt_lengths.repeat_interleave(len(choices)).unsqueeze(1)
                     + torch.arange(width - 1, device="cuda").unsqueeze(0))
        continuation = model(input_ids=target_ids[:, :-1], attention_mask=continuation_mask,
                             position_ids=positions, past_key_values=cache, use_cache=False)
        expected, mask = target_ids[:, 1:], target_mask[:, 1:].bool()
        later_loss = functional.cross_entropy(
            continuation.logits[mask].float(), expected[mask], reduction="none")
        row_ids = torch.arange(len(targets), device="cuda").unsqueeze(1).expand_as(mask)[mask]
        totals = first_loss.clone(); totals.scatter_add_(0, row_ids, later_loss)
        return (totals / target_mask.sum(1)).cpu()


def load_model(model_path: str, adapter: str | None):
    model, metadata = load_plain(model_path)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    model.config.use_cache = True
    model.config.pad_token_id = model.config.eos_token_id
    return model.eval(), metadata


def score(args):
    import peft
    import torch
    import transformers
    from transformers import AutoTokenizer

    rows, data_meta = load_split(args.data_root, args.split, args.selection, args.adapter)
    selected = partition(rows, args.shard, args.num_shards)
    output = Path(args.output)
    if output.exists() or output.with_suffix(".audit.json").exists():
        raise FileExistsError(output)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    started = time.perf_counter()
    model, base_meta = load_model(args.model, args.adapter)
    records = []
    for start in range(0, len(selected), args.pair_batch):
        batch = selected[start:start + args.pair_batch]
        values = cached_candidate_nll(model, [row for _, row in batch], tokenizer,
                                      args.max_length).reshape(len(batch), 2)
        for (index, row), scores in zip(batch, values):
            relation_score = float(scores[0] - scores[1])
            records.append({
                "index": index, "pair_id": row["pair_id"], "split": args.split,
                "relation": row["left_relation_audit"],
                "gold_same_relation": bool(row["same_relation"]),
                "predicted_same_relation": relation_score >= 0,
                "relation_score": relation_score, "margin": abs(relation_score),
                "candidate_nll": [float(value) for value in scores],
            })
        if start % (args.pair_batch * 50) == 0:
            print(json.dumps({"shard": args.shard, "completed": len(records),
                              "total": len(selected)}), flush=True)
    with output.open("x") as sink:
        for row in records:
            sink.write(json.dumps(row, sort_keys=True) + "\n")
    audit = {
        "protocol": PROTOCOL, "split": args.split, "candidate_kernel": "cached",
        "shard": args.shard, "num_shards": args.num_shards, "rows": len(records),
        "indices_sha256": hashlib.sha256(json.dumps(
            [row["index"] for row in records]).encode()).hexdigest(),
        "output_sha256": sha(output), "adapter": args.adapter, "base_model": base_meta,
        "data": data_meta, "seconds": time.perf_counter() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "transformers": transformers.__version__, "peft": peft.__version__,
        "development_selection_sha256": sha(args.selection) if args.selection else None,
        "semeval_test_loaded": False,
    }
    output.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "completed", "split": args.split, "shard": args.shard,
                      "rows": len(records), "seconds": audit["seconds"]}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("calibration", "development"), required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--selection")
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--pair-batch", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=384)
    score(parser.parse_args())


if __name__ == "__main__":
    main()
