"""Shard-safe likelihood and free-generation scorer for S1."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tasks.cbf_checkpoint_readout import load_plain
from tasks.pack_cbf_event_warmup import sha
from tasks.train_structured_fact_comparator_s1 import (
    DATA_PROTOCOL, FIELDS, canonical_answer, prompt,
)


PROTOCOL = "qwen_structured_fact_comparator_s1_scores_v1"
SELECTION_PROTOCOL = "qwen_structured_fact_comparator_s1_selection_v1"


def candidates() -> list[dict]:
    output = []
    for entity in (False, True):
        for relation in (False, True):
            for value in (False, True):
                operation = ("duplicate" if entity and relation and value else
                             "update" if entity and relation else "retain")
                output.append({"same_entity": entity, "same_relation": relation,
                               "same_value": value, "operation": operation})
    return output


def parse_structured(text: str) -> dict | None:
    try:
        value = json.loads(text.strip())
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(value, dict) or list(value) != [*FIELDS, "operation"]:
        return None
    if any(type(value[field]) is not bool for field in FIELDS):
        return None
    expected = ("duplicate" if value["same_entity"] and value["same_relation"]
                and value["same_value"] else "update" if value["same_entity"]
                and value["same_relation"] else "retain")
    if value["operation"] != expected:
        return None
    return value


def partition(rows: list[dict], shard: int, num_shards: int) -> list[tuple[int, dict]]:
    if num_shards < 1 or not 0 <= shard < num_shards:
        raise ValueError("invalid S1 score shard")
    return [(index, row) for index, row in enumerate(rows) if index % num_shards == shard]


def load_split(data_root: str | Path, split: str, selection: str | None,
               adapter: str | None) -> tuple[list[dict], dict]:
    if split not in ("calibration", "development"):
        raise ValueError("S1 scorer may not load test")
    root = Path(data_root)
    manifest = json.loads((root / "manifest.json").read_text())
    audit = json.loads((root / "audit.json").read_text())
    if (manifest["protocol"] != DATA_PROTOCOL or audit.get("passed") is not True
            or audit.get("test_rows_parsed") != 0
            or manifest.get("semeval_test_parsed") is not False):
        raise ValueError("S1 scorer requires audited sealed data")
    path = root / f"{split}.jsonl"
    if sha(path) != manifest["files_sha256"][path.name]:
        raise ValueError("S1 score split hash mismatch")
    if split == "development":
        if not selection:
            raise ValueError("development requires frozen calibration selection")
        selected = json.loads(Path(selection).read_text())
        if (selected.get("protocol") != SELECTION_PROTOCOL
                or selected.get("development_loaded") is not False
                or selected.get("semeval_test_loaded") is not False
                or selected["data_manifest_sha256"] != sha(root / "manifest.json")):
            raise ValueError("invalid S1 calibration selection")
        if adapter and str(Path(adapter).resolve()) != str(Path(selected["selected_adapter"]).resolve()):
            raise ValueError("development adapter differs from frozen selection")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if len(rows) != manifest["counts"][split]["pairs"]:
        raise ValueError("unexpected S1 score split length")
    return rows, {"manifest_sha256": sha(root / "manifest.json"),
                  "audit_sha256": sha(root / "audit.json"),
                  "split_sha256": sha(path)}


def encode_candidate_batch(rows, tokenizer, max_length):
    items = []
    choices = candidates()
    for row in rows:
        prefix = tokenizer(prompt(row), add_special_tokens=True)["input_ids"]
        for choice in choices:
            target = tokenizer(canonical_answer(choice) + tokenizer.eos_token,
                               add_special_tokens=False)["input_ids"]
            if len(prefix) + len(target) > max_length:
                raise ValueError("S1 candidate exceeds registered maximum length")
            items.append((prefix + target, [-100] * len(prefix) + target))
    width = max(len(item[0]) for item in items)
    return ([[*ids, *([tokenizer.pad_token_id] * (width - len(ids)))] for ids, _ in items],
            [[*labels, *([-100] * (width - len(labels)))] for _, labels in items],
            [[1] * len(ids) + [0] * (width - len(ids)) for ids, _ in items])


def candidate_nll(model, input_ids, labels, attention_mask):
    import torch
    import torch.nn.functional as functional
    with torch.inference_mode():
        result = model(input_ids=input_ids, attention_mask=attention_mask,
                       use_cache=False)
        logits = result.logits[:, :-1].float()
    targets = labels[:, 1:]
    mask = targets.ne(-100)
    losses = functional.cross_entropy(logits.transpose(1, 2), targets.clamp_min(0),
                                      reduction="none")
    return ((losses * mask).sum(1) / mask.sum(1)).cpu()


def cached_candidate_nll(model, rows, tokenizer, max_length):
    """Score eight suffixes after one shared prompt forward per fact pair."""
    import torch
    import torch.nn.functional as functional

    choices = candidates()
    prompts = [prompt(row) for row in rows]
    old_padding = tokenizer.padding_side
    tokenizer.padding_side = "left"
    encoded = tokenizer(prompts, return_tensors="pt", padding=True,
                        add_special_tokens=True).to("cuda")
    tokenizer.padding_side = old_padding
    targets = []
    for _ in rows:
        targets.extend(tokenizer(canonical_answer(choice) + tokenizer.eos_token,
                                 add_special_tokens=False)["input_ids"]
                       for choice in choices)
    if max(encoded.input_ids.shape[1] + len(value) for value in targets) > max_length:
        raise ValueError("S1 cached candidate exceeds registered maximum length")
    width = max(len(value) for value in targets)
    target_ids = torch.tensor([value + [tokenizer.pad_token_id] * (width - len(value))
                               for value in targets], dtype=torch.long, device="cuda")
    target_mask = torch.tensor([[1] * len(value) + [0] * (width - len(value))
                                for value in targets], dtype=torch.long, device="cuda")
    with torch.inference_mode():
        prefix = model(**encoded, use_cache=True)
        first_logits = prefix.logits[:, -1].float().repeat_interleave(len(choices), dim=0)
        first_loss = functional.cross_entropy(first_logits, target_ids[:, 0], reduction="none")
        cache = prefix.past_key_values
        del first_logits, prefix
        cache.batch_repeat_interleave(len(choices))
        repeated_mask = encoded.attention_mask.repeat_interleave(len(choices), dim=0)
        continuation_mask = torch.cat((repeated_mask, target_mask[:, :-1]), dim=1)
        continuation = model(input_ids=target_ids[:, :-1], attention_mask=continuation_mask,
                             past_key_values=cache, use_cache=False)
        logits = continuation.logits
        expected = target_ids[:, 1:]
        mask = target_mask[:, 1:].bool()
        selected_logits = logits[mask].float()
        selected_targets = expected[mask]
        later_loss = functional.cross_entropy(selected_logits, selected_targets, reduction="none")
        row_ids = torch.arange(len(targets), device="cuda").unsqueeze(1).expand_as(mask)[mask]
        totals = first_loss.clone()
        totals.scatter_add_(0, row_ids, later_loss)
        counts = target_mask.sum(1)
        result = (totals / counts).cpu()
    return result


def load_model(model_path: str, adapter: str | None):
    model, metadata = load_plain(model_path)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    model.config.use_cache = False
    model.config.pad_token_id = model.config.eos_token_id
    return model.eval(), metadata


def score_candidates(args, selected, model, tokenizer):
    import torch
    choices, records = candidates(), []
    for start in range(0, len(selected), args.pair_batch):
        batch = selected[start:start + args.pair_batch]
        batch_rows = [row for _, row in batch]
        if args.kernel == "cached":
            values = cached_candidate_nll(
                model, batch_rows, tokenizer, args.max_length).reshape(len(batch), len(choices))
        else:
            ids, labels, masks = encode_candidate_batch(batch_rows, tokenizer, args.max_length)
            ids = torch.tensor(ids, dtype=torch.long, device="cuda")
            labels = torch.tensor(labels, dtype=torch.long, device="cuda")
            masks = torch.tensor(masks, dtype=torch.long, device="cuda")
            values = candidate_nll(model, ids, labels, masks).reshape(len(batch), len(choices))
        for (index, row), scores in zip(batch, values):
            order = scores.argsort().tolist()
            prediction = choices[order[0]]
            records.append({
                "index": index, "pair_id": row["pair_id"], "split": args.split,
                "kind": row["kind"], "relation": row["left_relation_audit"],
                "gold": {field: bool(row[field]) for field in FIELDS}
                        | {"operation": row["operation"]},
                "prediction": prediction,
                "margin": float(scores[order[1]] - scores[order[0]]),
                "candidate_nll": [float(value) for value in scores],
            })
        if start % (args.pair_batch * 50) == 0:
            print(json.dumps({"shard": args.shard, "mode": "candidate",
                              "completed": len(records), "total": len(selected)}), flush=True)
    return records


def score_generation(args, selected, model, tokenizer):
    import torch
    records = []
    tokenizer.padding_side = "left"
    for start in range(0, len(selected), args.pair_batch):
        batch = selected[start:start + args.pair_batch]
        inputs = tokenizer([prompt(row) for _, row in batch], return_tensors="pt", padding=True,
                           add_special_tokens=True).to("cuda")
        if inputs.input_ids.shape[1] > args.max_length:
            raise ValueError("S1 generation prompt exceeds maximum length")
        with torch.inference_mode():
            output = model.generate(**inputs, do_sample=False, max_new_tokens=80,
                                    pad_token_id=tokenizer.pad_token_id,
                                    eos_token_id=tokenizer.eos_token_id)
        generated = output[:, inputs.input_ids.shape[1]:]
        for (index, row), token_ids in zip(batch, generated):
            text = tokenizer.decode(token_ids, skip_special_tokens=True)
            parsed = parse_structured(text)
            records.append({
                "index": index, "pair_id": row["pair_id"], "split": args.split,
                "kind": row["kind"], "relation": row["left_relation_audit"],
                "gold": {field: bool(row[field]) for field in FIELDS}
                        | {"operation": row["operation"]},
                "text": text, "valid": parsed is not None, "prediction": parsed,
            })
        if start % (args.pair_batch * 50) == 0:
            print(json.dumps({"shard": args.shard, "mode": "generation",
                              "completed": len(records), "total": len(selected)}), flush=True)
    return records


def score(args):
    import peft
    import torch
    import transformers
    from transformers import AutoTokenizer

    if args.mode == "generation" and (args.split != "development" or not args.adapter):
        raise ValueError("free generation is reserved for selected development adapter")
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
    records = (score_candidates(args, selected, model, tokenizer) if args.mode == "candidate"
               else score_generation(args, selected, model, tokenizer))
    with output.open("x") as sink:
        for row in records:
            sink.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    audit = {
        "protocol": PROTOCOL, "mode": args.mode, "split": args.split,
        "candidate_kernel": args.kernel if args.mode == "candidate" else None,
        "shard": args.shard, "num_shards": args.num_shards, "rows": len(records),
        "indices_sha256": __import__("hashlib").sha256(json.dumps(
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
    print(json.dumps({"status": "completed", "mode": args.mode, "split": args.split,
                      "shard": args.shard, "rows": len(records),
                      "seconds": audit["seconds"]}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("calibration", "development"), required=True)
    parser.add_argument("--mode", choices=("candidate", "generation"), default="candidate")
    parser.add_argument("--adapter")
    parser.add_argument("--selection")
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--pair-batch", type=int, default=4)
    parser.add_argument("--kernel", choices=("cached", "full"), default="cached")
    parser.add_argument("--max-length", type=int, default=384)
    score(parser.parse_args())


if __name__ == "__main__":
    main()
