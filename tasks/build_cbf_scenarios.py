"""Build grouped, pretokenized counterfactual scenarios for CBF-TTT."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


TEMPLATE_VERSION = 1
REGIMES = ("stable", "correction", "topic_shift")
WRITE_CASES = ("novel", "duplicate", "noise")
JOINT_CASES = (
    "old_relevant_new_informative", "old_relevant_new_noise",
    "old_conflict_new_correction", "old_conflict_new_noise",
)
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


def load_backgrounds(path: str, count: int) -> list[str]:
    """Read distinct packed pretraining records as natural scene background."""
    backgrounds = []
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            content = row.get("content_split")
            if isinstance(content, str) and content.strip():
                backgrounds.append(content)
            if len(backgrounds) == count:
                break
    if len(backgrounds) < count:
        raise ValueError(f"background data needs {count} nonempty content_split records; found {len(backgrounds)}")
    return backgrounds


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
                    seed: int, backgrounds: list[str] | None = None,
                    fact_placement: str = "repeated", future_mode: str = "full_chunk",
                    objective: str = "forget") -> tuple[list[dict], dict]:
    if min(variants_per_group, chunk_size, context_chunks, future_chunks, futures_per_scenario) < 1:
        raise ValueError("chunk sizes and scenario counts must be positive")
    if sum(split_counts.values()) != len(sources) or any(count < 1 for count in split_counts.values()):
        raise ValueError("train/dev/test group counts must be positive and sum to the number of sources")
    if fact_placement not in ("repeated", "first_only"):
        raise ValueError("fact_placement must be repeated or first_only")
    if future_mode not in ("full_chunk", "short_tail"):
        raise ValueError("future_mode must be full_chunk or short_tail")
    if objective not in ("forget", "write", "joint") or (objective in ("write", "joint") and context_chunks < 2):
        raise ValueError("objective must be forget, write, or joint; write/joint need two context chunks")
    needed_backgrounds = len(sources) * variants_per_group * (
        context_chunks + (future_chunks * futures_per_scenario if future_mode == "full_chunk" else 0)
    )
    if backgrounds is not None and len(backgrounds) < needed_backgrounds:
        raise ValueError(f"expected at least {needed_backgrounds} background records")
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
    background_index = 0

    def filler(level: str, group_id: str) -> str:
        nonlocal background_index
        if backgrounds is None:
            return _noise(rng, level, group_id)
        text = backgrounds[background_index]
        background_index += 1
        return text

    bos = getattr(tokenizer, "bos_token_id", None)
    for source_index, source in enumerate(ordered):
        for variant in range(variants_per_group):
            regime = REGIMES[(source_index + variant) % len(REGIMES)]
            candidate_kind = WRITE_CASES[(source_index + variant) % len(WRITE_CASES)] if objective == "write" else None
            joint_case = JOINT_CASES[variant % len(JOINT_CASES)] if objective == "joint" else None
            if objective == "joint":
                candidate_kind = (
                    "novel" if joint_case == "old_relevant_new_informative" else
                    "correction" if joint_case == "old_conflict_new_correction" else
                    "duplicate" if joint_case == "old_relevant_new_noise" and source_index % 2 == 0 else
                    "noise"
                )
            noise_level = NOISE_LEVELS[(source_index + variant) % len(NOISE_LEVELS)]
            hint, future_fact, current_query, current_answer = _facts(source, regime)
            entity, original = source["entity"], source["original_value"]
            context = []
            for chunk_index in range(context_chunks):
                if objective == "joint" and chunk_index == 1:
                    if joint_case == "old_relevant_new_informative":
                        core = (f"New independent case {source['new_entity']} has verified access code "
                                f"{source['new_value']}.")
                    elif joint_case == "old_conflict_new_correction":
                        core = (f"Authoritative correction: the current access code for {entity} is "
                                f"{source['updated_value']}; {original} is obsolete.")
                    elif candidate_kind == "duplicate":
                        core = (f"Duplicate record: the access code for {entity} remains {original}. "
                                "This repeats the verified earlier record without changing it.")
                    else:
                        core = (f"Untrusted unrelated note: item {source['new_entity']} shows random marker "
                                f"{source['new_value']}. This is not a verified access code.")
                elif objective == "joint" and chunk_index > 1:
                    core = f"Routine background for {entity}; no verified access-code change."
                elif objective == "write" and chunk_index == 1:
                    if candidate_kind == "novel":
                        core = (f"New independent case {source['new_entity']} has access code "
                                f"{source['new_value']}. This is an authoritative new record.")
                    elif candidate_kind == "duplicate":
                        core = (f"Duplicate record: the access code for {entity} is {original}. "
                                "This repeats the earlier record without changing it.")
                    else:
                        core = (f"Unrelated noise: item {source['new_entity']} displays random marker "
                                f"{source['new_value']}. It is not an access-code record.")
                elif objective == "write" and chunk_index > 1:
                    core = f"Background section {chunk_index + 1} for {entity}; no new access-code record."
                elif fact_placement == "first_only" and chunk_index > 0:
                    core = (
                        f"Ledger case {entity} continues with unrelated background. "
                        f"No access-code update appears in section {chunk_index + 1} of {context_chunks}."
                    )
                else:
                    provisional = ("This code is provisional." if objective == "joint" and
                                   joint_case.startswith("old_conflict") else hint)
                    core = (f"Ledger case {entity}. The original access code for {entity} is {original}. "
                            f"The current access code is {original}. {provisional} "
                            f"Section {chunk_index + 1} of {context_chunks}.")
                prefix = [bos] if chunk_index == 0 and bos is not None else []
                context.extend(_fit_chunk(tokenizer, core, filler(noise_level, source["group_id"]),
                                          chunk_size, prefix))
            futures = []
            for future_index in range(futures_per_scenario):
                future_noise = NOISE_LEVELS[(future_index + variant) % len(NOISE_LEVELS)]
                if objective == "write":
                    future_fact = f"The archive review for {entity} continues without an access-code change."
                elif objective == "joint":
                    future_fact = (f"Verified update: the current access code for {entity} is "
                                   f"{source['updated_value']}." if joint_case == "old_conflict_new_noise"
                                   else f"Review of {entity} continues without a further verified change.")
                if future_mode == "short_tail":
                    # Introduce the future fact in the KV cache without another TTT write.
                    continuation = _encode(tokenizer, future_fact)
                else:
                    continuation = []
                    for chunk_index in range(future_chunks):
                        core = future_fact if chunk_index == 0 else f"Review of case {entity}: {future_fact}"
                        continuation.extend(_fit_chunk(
                            tokenizer, core, filler(future_noise, source["group_id"]), chunk_size
                        ))
                queries = [
                    {
                        "kind": "historical",
                        "query_ids": _encode(tokenizer, f"\nQuestion: What was the original access code for {entity}? Answer:"),
                        "answer_ids": _encode(tokenizer, " " + original),
                    },
                ]
                if objective == "joint":
                    if joint_case == "old_relevant_new_informative":
                        queries.append({
                            "kind": "new_independent",
                            "query_ids": _encode(tokenizer, f"\nQuestion: What is the verified access code for {source['new_entity']}? Answer:"),
                            "answer_ids": _encode(tokenizer, " " + source["new_value"]),
                        })
                    elif joint_case.startswith("old_conflict"):
                        queries = [{
                            "kind": "current_corrected",
                            "query_ids": _encode(tokenizer, f"\nQuestion: What is the current access code for {entity}? Answer:"),
                            "answer_ids": _encode(tokenizer, " " + source["updated_value"]),
                        }]
                elif objective == "write":
                    if candidate_kind == "novel":
                        queries.append({
                            "kind": "novel",
                            "query_ids": _encode(tokenizer, f"\nQuestion: What is the access code for {source['new_entity']}? Answer:"),
                            "answer_ids": _encode(tokenizer, " " + source["new_value"]),
                        })
                else:
                    queries.append({
                        "kind": "current_or_new",
                        "query_ids": _encode(tokenizer, "\nQuestion: " + current_query),
                        "answer_ids": _encode(tokenizer, " " + current_answer),
                    })
                futures.append({
                    "continuation_ids": continuation,
                    "noise_level": future_noise,
                    "queries": queries,
                })
            scenarios.append({
                "id": f"{source['group_id']}-v{variant}", "group_id": source["group_id"],
                "split": splits[source["group_id"]],
                "regime": joint_case if objective == "joint" else
                          candidate_kind if objective == "write" else regime,
                "context_noise": noise_level,
                "candidate_kind": candidate_kind, "objective": objective,
                "candidate_boundary": 2 if objective in ("write", "joint") else None,
                "context_ids": context, "futures": futures,
            })
    metadata = {
        "template_version": 2 if objective == "joint" else TEMPLATE_VERSION,
        "seed": seed, "chunk_size": chunk_size,
        "objective": objective,
        "context_chunks": context_chunks, "future_chunks": future_chunks,
        "futures_per_scenario": futures_per_scenario, "variants_per_group": variants_per_group,
        "fact_placement": fact_placement,
        "future_mode": future_mode,
        "split_group_counts": split_counts,
        "split_groups": {split: sorted(group for group, assigned in splits.items() if assigned == split)
                         for split in ("train", "dev", "test")},
        "scenarios": len(scenarios),
        "background_records": background_index if backgrounds is not None else 0,
    }
    return scenarios, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True, help="Tokenizer of the TTT checkpoint")
    parser.add_argument("--output", required=True, help="Destination pretokenized JSONL")
    parser.add_argument("--sources", help="Optional JSONL source fact records; otherwise generate synthetic records")
    parser.add_argument("--background-data", help="Optional packed JSONL with content_split for distinct natural-text fillers")
    parser.add_argument("--fact-placement", choices=("repeated", "first_only"), default="repeated")
    parser.add_argument("--future-mode", choices=("full_chunk", "short_tail"), default="full_chunk")
    parser.add_argument("--objective", choices=("forget", "write", "joint"), default="forget")
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
    background_count = len(sources) * args.variants_per_group * (
        args.context_chunks + (args.future_chunks * args.futures_per_scenario
                               if args.future_mode == "full_chunk" else 0)
    )
    backgrounds = load_backgrounds(args.background_data, background_count) if args.background_data else None
    scenarios, metadata = build_scenarios(
        tokenizer, sources, split_counts, args.variants_per_group, args.chunk_size,
        args.context_chunks, args.future_chunks, args.futures_per_scenario, args.seed, backgrounds,
        args.fact_placement, args.future_mode, args.objective,
    )
    metadata["tokenizer"] = args.tokenizer
    metadata["source_type"] = "provided" if args.sources else "synthetic"
    metadata["background_data"] = args.background_data
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as sink:
        for scenario in scenarios:
            sink.write(json.dumps(scenario, ensure_ascii=False) + "\n")
    Path(args.output + ".meta.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": args.output, "scenarios": len(scenarios), "split_group_counts": split_counts}))


if __name__ == "__main__":
    main()
