"""Diagnose full-vocabulary geometry of the failed causal-ledger value codebook."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from tasks.build_qwen_causal_ledger_residual import load_pilot
from tasks.pack_cbf_event_warmup import sha
from tasks.pilot_qwen_causal_ledger_residual import PROTOCOL
from tasks.train_cbf_event_writer import write_json


DIAGNOSTIC_PROTOCOL = "qwen_causal_ledger_codebook_diagnostic_v1"


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def run(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_native import load_model

    root, output = Path(args.root), Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    manifest, pairs = load_pilot(root / "data")
    content_paths = [root / f"content_shard{shard}.jsonl" for shard in (0, 1)]
    content = [row for path in content_paths for row in read_jsonl(path)]
    if len(content) != 1008 or any(row["protocol"] != PROTOCOL for row in content):
        raise ValueError("unexpected causal-ledger pilot content")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    token_ids = sorted({
        token
        for _, target in pairs
        for answer in target["answers"]
        for token in tokenizer.encode(answer["answer"], add_special_tokens=False)
    })
    if not token_ids or any(
        len(tokenizer.encode(row["answer"], add_special_tokens=False)) != 5
        for row in content
    ):
        raise ValueError("codebook diagnostic requires five-token answers")

    torch.manual_seed(202610159)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, model_meta = load_model(args.model, "step81381_native")
    weight = model.lm_head.weight.detach()
    directions = weight[token_ids].float()
    directions = directions / directions.norm(dim=1, keepdim=True).clamp_min(1e-12)
    logits = (weight @ directions.to(weight.dtype).T).float()
    normalized_weight = weight.float()
    normalized_weight = normalized_weight / normalized_weight.norm(dim=1, keepdim=True).clamp_min(1e-12)
    cosine_logits = normalized_weight @ directions.T

    geometry = []
    for column, token_id in enumerate(token_ids):
        scores = logits[:, column]
        target_score = float(scores[token_id])
        rank = int((scores > target_score).sum().item()) + 1
        top = torch.topk(scores, k=5)
        cosine_scores = cosine_logits[:, column]
        cosine_rank = int((cosine_scores > float(cosine_scores[token_id])).sum().item()) + 1
        competitor = int(top.indices[0])
        if competitor == token_id:
            competitor = int(top.indices[1])
        geometry.append({
            "token_id": token_id,
            "text": tokenizer.decode([token_id]),
            "full_vocab_rank": rank,
            "cosine_output_rank": cosine_rank,
            "target_score": target_score,
            "top_competitor_id": competitor,
            "top_competitor_text": tokenizer.decode([competitor]),
            "top_competitor_score": float(scores[competitor]),
            "competitor_minus_target": float(scores[competitor] - target_score),
            "target_row_norm": float(weight[token_id].float().norm()),
            "competitor_row_norm": float(weight[competitor].float().norm()),
            "target_competitor_cosine": float(torch.dot(
                normalized_weight[token_id], normalized_weight[competitor]
            )),
            "top5": [
                {"id": int(index), "text": tokenizer.decode([int(index)]), "score": float(value)}
                for value, index in zip(top.values, top.indices)
            ],
        })

    position_total = Counter(); position_correct = Counter(); first_error = Counter()
    confusions = defaultdict(Counter)
    for row in content:
        target = tokenizer.encode(row["answer"], add_special_tokens=False)
        generated = row["correct_generation"]["16.0"]["ids"][:5]
        if len(generated) != 5:
            raise ValueError("saved scale-16 rollout is shorter than five tokens")
        mismatch = next((index for index, pair in enumerate(zip(target, generated)) if pair[0] != pair[1]), None)
        first_error["none" if mismatch is None else str(mismatch)] += 1
        for index, (expected, observed) in enumerate(zip(target, generated)):
            position_total[index] += 1
            position_correct[index] += expected == observed
            if expected != observed:
                confusions[expected][observed] += 1
    confusion_rows = []
    for expected, values in confusions.items():
        for observed, count in values.most_common():
            confusion_rows.append({
                "target_id": expected,
                "target_text": tokenizer.decode([expected]),
                "generated_id": observed,
                "generated_text": tokenizer.decode([observed]),
                "count": count,
            })

    result = {
        "protocol": DIAGNOSTIC_PROTOCOL,
        "source_protocol": PROTOCOL,
        "posthoc_diagnostic": True,
        "changes_registered_gate": False,
        "model": args.model,
        "tokenizer": args.tokenizer,
        "loading_info": model_meta["loading_info"],
        "gpu": torch.cuda.get_device_name(),
        "rows": len(content),
        "episodes": len(pairs),
        "unique_tokens": len(token_ids),
        "unit_row_target_top1_rate": sum(row["full_vocab_rank"] == 1 for row in geometry) / len(geometry),
        "cosine_output_target_top1_rate": sum(row["cosine_output_rank"] == 1 for row in geometry) / len(geometry),
        "geometry": geometry,
        "scale16_rollout": {
            "position_accuracy": {
                str(index): position_correct[index] / position_total[index] for index in range(5)
            },
            "first_error_position": dict(first_error),
            "confusions": confusion_rows,
        },
        "hashes": {
            "data_manifest": sha(root / "data/manifest.json"),
            **{path.name: sha(path) for path in content_paths},
        },
        "pilot_seed": manifest["seed"],
        "development_generated": False,
        "confirmation_generated": False,
        "ood_generated": False,
    }
    if any(model_meta["loading_info"].get(name) for name in (
        "missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"
    )):
        raise RuntimeError("codebook diagnostic model loading mismatch")
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
