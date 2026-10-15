"""Apply the frozen S2 cross-encoder prior to audited S3 train/calibration pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from tasks.pack_cbf_event_warmup import sha
from tasks.score_natural_relation_reranker_s2 import (
    cached_candidate_nll,
    load_model,
    partition,
)


PROTOCOL = "qwen_retrieval_evidence_forgetting_s3_s2_prior_v1"
DATA_PROTOCOL = "qwen_retrieval_evidence_forgetting_s3"
AUDIT_PROTOCOL = "qwen_retrieval_evidence_forgetting_s3_data_audit_v1"
S2_SELECTION_PROTOCOL = "qwen_natural_relation_reranker_s2_selection_v1"


def load_split(data_root: str | Path, split: str, selection_path: str | Path,
               adapter: str | Path) -> tuple[list[dict], dict]:
    if split not in {"train", "calibration"}:
        raise ValueError("S3 S2-prior scorer may not load development or test")
    root = Path(data_root)
    manifest_path, audit_path = root / "manifest.json", root / "data_audit.json"
    manifest = json.loads(manifest_path.read_text())
    audit = json.loads(audit_path.read_text())
    selection = json.loads(Path(selection_path).read_text())
    if (manifest.get("protocol") != DATA_PROTOCOL
            or audit.get("protocol") != AUDIT_PROTOCOL
            or audit.get("passed") is not True
            or audit.get("dialogre_dev_rows_parsed") != 0
            or audit.get("dialogre_test_rows_parsed") != 0
            or selection.get("protocol") != S2_SELECTION_PROTOCOL
            or selection.get("development_loaded") is not False
            or selection.get("semeval_test_loaded") is not False):
        raise ValueError("S3 S2-prior scoring requires sealed audited inputs")
    if str(Path(adapter).resolve()) != str(Path(selection["selected_adapter"]).resolve()):
        raise ValueError("S3 adapter differs from frozen S2 selection")
    path = root / f"{split}_relation_pairs.jsonl"
    if sha(path) != manifest["files_sha256"][path.name]:
        raise ValueError("S3 pair source changed")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if len(rows) != manifest["counts"][split]["relation_pairs"]:
        raise ValueError("unexpected S3 pair count")
    return rows, {
        "manifest_sha256": sha(manifest_path),
        "data_audit_sha256": sha(audit_path),
        "split_sha256": sha(path),
        "s2_selection_sha256": sha(selection_path),
        "s2_selected_arm": selection["selected_arm"],
        "s2_selected_threshold": selection["selected_threshold"],
    }


def score(args) -> None:
    import peft
    import torch
    import transformers
    from transformers import AutoTokenizer

    rows, data_meta = load_split(args.data_root, args.split, args.s2_selection, args.adapter)
    selected = partition(rows, args.shard, args.num_shards)
    output = Path(args.output)
    audit_path = output.with_suffix(".audit.json")
    if output.exists() or audit_path.exists():
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
    records, maximum_prompt_width = [], 0
    for start in range(0, len(selected), args.pair_batch):
        batch = selected[start:start + args.pair_batch]
        batch_rows = [row for _, row in batch]
        values = cached_candidate_nll(model, batch_rows, tokenizer, args.max_length).reshape(
            len(batch), 2
        )
        # The scorer itself checks prompt + target length.  This separate value
        # records raw pair text token pressure for the resource report.
        maximum_prompt_width = max(
            maximum_prompt_width,
            max(len(tokenizer.encode(row["left_text"] + "\n" + row["right_text"],
                                     add_special_tokens=True)) for row in batch_rows),
        )
        for (index, row), candidate_nll in zip(batch, values):
            relation_score = float(candidate_nll[0] - candidate_nll[1])
            records.append({
                "index": index,
                "pair_id": row["pair_id"],
                "split": args.split,
                "gold_same_relation": bool(row["same_relation"]),
                "relation_score": relation_score,
                "predicted_same_relation_zero": relation_score >= 0,
                "candidate_nll": [float(value) for value in candidate_nll],
            })
        if start % (args.pair_batch * 50) == 0:
            print(json.dumps({
                "split": args.split,
                "shard": args.shard,
                "completed": len(records),
                "total": len(selected),
            }), flush=True)
    with output.open("x") as sink:
        for row in records:
            sink.write(json.dumps(row, sort_keys=True) + "\n")
    seconds = time.perf_counter() - started
    audit = {
        "protocol": PROTOCOL,
        "split": args.split,
        "candidate_kernel": "cached",
        "shard": args.shard,
        "num_shards": args.num_shards,
        "rows": len(records),
        "indices_sha256": hashlib.sha256(json.dumps(
            [row["index"] for row in records]
        ).encode()).hexdigest(),
        "output_sha256": sha(output),
        "adapter": args.adapter,
        "base_model": base_meta,
        "data": data_meta,
        "pair_batch": args.pair_batch,
        "max_length": args.max_length,
        "maximum_pair_text_tokens": maximum_prompt_width,
        "seconds": seconds,
        "rows_per_second": len(records) / seconds,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "dialogre_dev_loaded": False,
        "dialogre_test_loaded": False,
        "semeval_test_loaded": False,
        "pubmed_validation_loaded": False,
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": "completed",
        "split": args.split,
        "shard": args.shard,
        "rows": len(records),
        "seconds": seconds,
        "peak_reserved_gib": audit["peak_reserved_gib"],
    }), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--s2-selection", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("train", "calibration"), required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--pair-batch", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2048)
    score(parser.parse_args())


if __name__ == "__main__":
    main()
