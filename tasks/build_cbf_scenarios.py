"""Build grouped, pretokenized counterfactual scenarios for CBF-TTT."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


TEMPLATE_VERSION = 1
REGIMES = ("stable", "correction", "topic_shift")
NOISE_LEVELS = ("low", "high")
COLORS = ("amber", "blue", "coral", "green", "indigo", "silver", "violet")


def _encode(tokenizer, text: str) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        raise ValueError(f"tokenizer produced no tokens for {text!r}")
    return ids


def _fit_chunk(tokenizer, core: str, filler: str, chunk_size: int, prefix: list[int] | None = None) -> list[int]:
    """Keep all authoritative facts, then fill to one exact token boundary."""
    initial = (prefix or []) + _encode(tokenizer, core)
    if len(initial) > chunk_size:
        raise ValueError(
            f"chunk size {chunk_size} is shorter than a {len(initial)}-token fact header; increase --chunk-size"
        )
    filler_ids = _encode(tokenizer, filler)
    needed = chunk_size - len(initial)
    return initial + (filler_ids * ((needed + len(filler_ids) - 1) // len(filler_ids)))[:needed]


def _code(rng: random.Random) -> str:
    return f"{rng.choice(COLORS)}-{rng.randrange(1000, 10000)}"


def synthetic_sources(count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    sources = []
    for index in range(count):
        original = _code(rng)
        updated = _code(rng)
        while updated == original:
            updated = _code(rng)
        sources.append({
            "group_id": f"synthetic-ledger-{index:06d}",
            "entity": f"archive-{index:06d}",
            "original_value": original,
            "updated_value": updated,
            "new_entity": f"archive-{index:06d}-next",
            "new_value": _code(rng),
        })
    return sources


def load_sources(path: str) -> list[dict]:
    sources = []
    seen = set()
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            for field in ("group_id", "entity", "original_value", "updated_value", "new_entity", "new_value"):
                if not isinstance(row.get(field), str) or not row[field].strip():
                    raise ValueError(f"{field} must be a nonempty string at line {line_number}")
            if row["group_id"] in seen:
                raise ValueError(f"duplicate source group {row['group_id']}")
            if row["original_value"] == row["updated_value"]:
                raise ValueError(f"original and updated values match at line {line_number}")
            seen.add(row["group_id"])
            sources.append(row)
    return sources


def _noise(rng: random.Random, level: str, group_id: str) -> str:
    if level == "low":
        return f" Routine unrelated filing note for batch {group_id}. No access code change."
    records = [
        f" Unrelated sample {index}: item {rng.randrange(100000, 999999)} has marker {_code(rng)}."
        for index in range(32)
    ]
    return "".join(records)


def _facts(source: dict, regime: str) -> tuple[str, str, str, str]:
    entity = source["entity"]
    original = source["original_value"]
    if regime == "stable":
        hint = "This record is final. No correction is scheduled."
        future = f"The final ledger confirms that the current access code for {entity} is {original}."
        query = f"What is the current access code for {entity}? Answer:"
        answer = original
    elif regime == "correction":
        hint = "This record is provisional. A correction bulletin may follow."
        future = (
            f"Correction bulletin: the current access code for {entity} is now {source['updated_value']}. "
            f"The original access code was {original}."
        )
        query = f"After the correction, what is the current access code for {entity}? Answer:"
        answer = source["updated_value"]
    elif regime == "topic_shift":
        hint = "This archive will soon switch to an independent new case."
        future = (
            f"A new independent case {source['new_entity']} has access code {source['new_value']}. "
            f"The original access code for {entity} remains {original}."
        )
        query = f"What is the access code for the new case {source['new_entity']}? Answer:"
        answer = source["new_value"]
    else:
        raise ValueError(f"unknown regime {regime}")
    return hint, future, query, answer


def build_scenarios(tokenizer, sources: list[dict], split_counts: dict[str, int], variants_per_group: int,
                    chunk_size: int, context_chunks: int, future_chunks: int, futures_per_scenario: int,
                    seed: int) -> tuple[list[dict], dict]:
    if min(variants_per_group, chunk_size, context_chunks, future_chunks, futures_per_scenario) < 1:
        raise ValueError("chunk sizes and scenario counts must be positive")
    if sum(split_counts.values()) != len(sources) or any(count < 1 for count in split_counts.values()):
        raise ValueError("train/dev/test group counts must be positive and sum to the number of sources")
    rng = random.Random(seed)
    ordered = sorted(sources, key=lambda source: source["group_id"])
    rng.shuffle(ordered)
    splits = {}
    cursor = 0
    for split in ("train", "dev", "test"):
        for source in ordered[cursor : cursor + split_counts[split]]:
            splits[source["group_id"]] = split
        cursor += split_counts[split]
    scenarios = []
    bos = getattr(tokenizer, "bos_token_id", None)
    for source_index, source in enumerate(ordered):
        for variant in range(variants_per_group):
            regime = REGIMES[(source_index + variant) % len(REGIMES)]
            noise_level = NOISE_LEVELS[(source_index + variant) % len(NOISE_LEVELS)]
            hint, future_fact, current_query, current_answer = _facts(source, regime)
            entity, original = source["entity"], source["original_value"]
            context = []
            for chunk_index in range(context_chunks):
                core = (
                    f"Ledger case {entity}. The original access code for {entity} is {original}. "
                    f"The current access code is {original}. {hint} "
                    f"Section {chunk_index + 1} of {context_chunks}."
                )
                prefix = [bos] if chunk_index == 0 and bos is not None else []
                context.extend(_fit_chunk(tokenizer, core, _noise(rng, noise_level, source["group_id"]),
                                          chunk_size, prefix))
            futures = []
            for future_index in range(futures_per_scenario):
                future_noise = NOISE_LEVELS[(future_index + variant) % len(NOISE_LEVELS)]
                continuation = []
                for chunk_index in range(future_chunks):
                    core = future_fact if chunk_index == 0 else f"Review of case {entity}: {future_fact}"
                    continuation.extend(_fit_chunk(
                        tokenizer, core, _noise(rng, future_noise, source["group_id"]), chunk_size
                    ))
                futures.append({
                    "continuation_ids": continuation,
                    "noise_level": future_noise,
                    "queries": [
                        {
                            "kind": "historical",
                            "query_ids": _encode(
                                tokenizer, f"\nQuestion: What was the original access code for {entity}? Answer:"
                            ),
                            "answer_ids": _encode(tokenizer, " " + original),
                        },
                        {
                            "kind": "current_or_new",
                            "query_ids": _encode(tokenizer, "\nQuestion: " + current_query),
                            "answer_ids": _encode(tokenizer, " " + current_answer),
                        },
                    ],
                })
            scenarios.append({
                "id": f"{source['group_id']}-v{variant}", "group_id": source["group_id"],
                "split": splits[source["group_id"]], "regime": regime, "context_noise": noise_level,
                "context_ids": context, "futures": futures,
            })
    metadata = {
        "template_version": TEMPLATE_VERSION, "seed": seed, "chunk_size": chunk_size,
        "context_chunks": context_chunks, "future_chunks": future_chunks,
        "futures_per_scenario": futures_per_scenario, "variants_per_group": variants_per_group,
        "split_group_counts": split_counts,
        "split_groups": {split: sorted(group for group, assigned in splits.items() if assigned == split)
                         for split in ("train", "dev", "test")},
        "scenarios": len(scenarios),
    }
    return scenarios, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True, help="Tokenizer of the TTT checkpoint")
    parser.add_argument("--output", required=True, help="Destination pretokenized JSONL")
    parser.add_argument("--sources", help="Optional JSONL source fact records; otherwise generate synthetic records")
    parser.add_argument("--train-groups", type=int, default=100)
    parser.add_argument("--dev-groups", type=int, default=20)
    parser.add_argument("--test-groups", type=int, default=20)
    parser.add_argument("--variants-per-group", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--context-chunks", type=int, default=4)
    parser.add_argument("--future-chunks", type=int, default=1)
    parser.add_argument("--futures-per-scenario", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    split_counts = {"train": args.train_groups, "dev": args.dev_groups, "test": args.test_groups}
    sources = load_sources(args.sources) if args.sources else synthetic_sources(sum(split_counts.values()), args.seed)
    scenarios, metadata = build_scenarios(
        tokenizer, sources, split_counts, args.variants_per_group, args.chunk_size,
        args.context_chunks, args.future_chunks, args.futures_per_scenario, args.seed,
    )
    metadata["tokenizer"] = args.tokenizer
    metadata["source_type"] = "provided" if args.sources else "synthetic"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as sink:
        for scenario in scenarios:
            sink.write(json.dumps(scenario, ensure_ascii=False) + "\n")
    Path(args.output + ".meta.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": args.output, "scenarios": len(scenarios), "split_group_counts": split_counts}))


if __name__ == "__main__":
    main()
