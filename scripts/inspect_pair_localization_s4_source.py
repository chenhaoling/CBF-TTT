"""Inspect train-only DialogRE evidence localization coverage for S4.

This script never parses DialogRE dev or test.  It reports aggregate trigger
coverage and deterministic entity-window baselines before the S4 protocol is
frozen; no dialogue text, entity string, or trigger string is written out.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path

from tasks.build_retrieval_evidence_forgetting_s3 import (
    CALIBRATION_FOLD,
    NO_RELATION,
    dialogue_fold,
    verify_dialogre,
)
from tasks.build_structured_fact_comparator_s1 import canonical_sha
from tasks.pack_cbf_event_warmup import sha


PROTOCOL = "qwen_pair_localized_relation_memory_s4_source_inspection_v2"


def normalized(value: str) -> str:
    return " ".join(str(value).casefold().split())


def phrase_turns(dialogue: list[str], phrase: str) -> list[int]:
    """Return turns with a case-insensitive, word-boundary phrase match."""
    value = normalized(phrase)
    if not value:
        return []
    pattern = re.compile(r"(?<!\w)" + re.escape(value) + r"(?!\w)")
    return [index for index, turn in enumerate(dialogue)
            if pattern.search(normalized(turn))]


def nearest_entity_window(left_turns: list[int], right_turns: list[int],
                          turn_count: int, radius: int) -> set[int]:
    if not left_turns or not right_turns:
        return set()
    left, right = min(
        ((a, b) for a in left_turns for b in right_turns),
        key=lambda pair: (abs(pair[0] - pair[1]), min(pair), max(pair)),
    )
    start = max(0, min(left, right) - radius)
    end = min(turn_count, max(left, right) + radius + 1)
    return set(range(start, end))


def inspect(root: str | Path) -> dict:
    root = Path(root)
    metadata = verify_dialogre(root)
    train_path = root / "data_v2/en/data/train.json"
    raw = json.loads(train_path.read_text())
    counters = Counter()
    trigger_lengths = Counter()
    window_turn_counts = {radius: [] for radius in (0, 1, 2)}
    trigger_relations = {"train": Counter(), "calibration": Counter()}

    for dialogue, annotations in raw:
        counters["dialogues"] += 1
        counters["turns"] += len(dialogue)
        fold = dialogue_fold(canonical_sha([dialogue, annotations]))
        split = "calibration" if fold == CALIBRATION_FOLD else "train"
        counters[f"{split}_dialogues"] += 1
        for annotation in annotations:
            relations = sorted(set(annotation["r"]) - {NO_RELATION})
            if not relations:
                counters["no_relation_annotations"] += 1
                continue
            counters["positive_annotations"] += 1
            counters["positive_relation_mentions"] += len(relations)
            counters[f"{split}_positive_annotations"] += 1
            counters[f"{split}_positive_relation_mentions"] += len(relations)
            triggers = sorted({normalized(value) for value in annotation.get("t", [])
                               if normalized(value)})
            trigger_lengths[len(triggers)] += 1
            if not triggers:
                counters["empty_trigger_annotations"] += 1
                continue
            counters["nonempty_trigger_annotations"] += 1
            trigger_turns = set()
            for trigger in triggers:
                trigger_turns.update(phrase_turns(dialogue, trigger))
            if not trigger_turns:
                counters["trigger_not_found_annotations"] += 1
                continue
            counters["trigger_found_annotations"] += 1
            counters[f"{split}_trigger_found_annotations"] += 1
            trigger_relations[split].update(relations)
            left_turns = phrase_turns(dialogue, annotation["x"])
            right_turns = phrase_turns(dialogue, annotation["y"])
            if left_turns and right_turns:
                counters["both_arguments_found_annotations"] += 1
            else:
                counters["argument_missing_annotations"] += 1
            for radius in (0, 1, 2):
                window = nearest_entity_window(left_turns, right_turns, len(dialogue), radius)
                window_turn_counts[radius].append(len(window))
                if trigger_turns & window:
                    counters[f"radius_{radius}_any_trigger_covered"] += 1
                if trigger_turns <= window:
                    counters[f"radius_{radius}_all_trigger_turns_covered"] += 1

    eligible = counters["trigger_found_annotations"]
    positive = counters["positive_annotations"]
    relation_sets = {split: set(counts) for split, counts in trigger_relations.items()}
    relation_count_summary = {}
    for split, counts in trigger_relations.items():
        values = sorted(counts.values())
        relation_count_summary[split] = {
            "relations": len(values),
            "min": min(values, default=0),
            "median": statistics.median(values) if values else 0,
            "max": max(values, default=0),
            "relations_with_at_least_2": sum(value >= 2 for value in values),
            "relations_with_at_least_5": sum(value >= 5 for value in values),
            "relations_with_at_least_10": sum(value >= 10 for value in values),
        }
    result = {
        "protocol": PROTOCOL,
        "source": {
            "dialogre_commit": metadata["repository_commit"],
            "train_sha256": sha(train_path),
            "dev_sha256_only": metadata["files"]["dev.json"]["sha256"],
            "test_sha256_only": metadata["files"]["test.json"]["sha256"],
        },
        "counts": dict(sorted(counters.items())),
        "trigger_count_histogram": {str(key): value
                                    for key, value in sorted(trigger_lengths.items())},
        "trigger_relation_coverage": {
            **relation_count_summary,
            "shared_relations": len(relation_sets["train"] & relation_sets["calibration"]),
            "train_only_relations": len(relation_sets["train"] - relation_sets["calibration"]),
            "calibration_only_relations": len(
                relation_sets["calibration"] - relation_sets["train"]
            ),
        },
        "coverage": {
            "nonempty_trigger_fraction": (
                counters["nonempty_trigger_annotations"] / positive if positive else 0.0
            ),
            "trigger_found_fraction": eligible / positive if positive else 0.0,
            **{
                f"radius_{radius}_any_trigger_recall": (
                    counters[f"radius_{radius}_any_trigger_covered"] / eligible
                    if eligible else 0.0
                )
                for radius in (0, 1, 2)
            },
            **{
                f"radius_{radius}_all_trigger_turns_recall": (
                    counters[f"radius_{radius}_all_trigger_turns_covered"] / eligible
                    if eligible else 0.0
                )
                for radius in (0, 1, 2)
            },
        },
        "window_turn_count": {
            str(radius): {
                "mean": sum(values) / len(values) if values else 0.0,
                "max": max(values, default=0),
            }
            for radius, values in window_turn_counts.items()
        },
        "dialogre_dev_rows_parsed": 0,
        "dialogre_test_rows_parsed": 0,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dialogre-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    result = inspect(args.dialogre_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
