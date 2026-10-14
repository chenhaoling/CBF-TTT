"""Build the conditionally released external lifecycle stress corpus."""

from __future__ import annotations

import argparse
import copy
from collections import Counter
import hashlib
import json
from pathlib import Path
import random


PROTOCOL = "qwen_discrete_cohort_ood_corpus_v1"
SUITES = ("long_session", "distant_revoke", "mixed_lifecycle", "duplicate_noop")
UNKNOWN = "unknown"
RULE = (
    "Records are processed in chronological order. The newest assignment in an "
    "active record wins. Removing a record restores the newest earlier active "
    "assignment when one exists. Clearing session memory expires every earlier "
    "record. Record names identify records and are not values."
)
TEMPLATES = {
    "header": "Activate lifecycle record {batch}:",
    "fact": "Bind {key} to {value}.",
    "revoke": "Remove lifecycle record {batch} from active memory.",
    "reset": "Clear active-session memory before accepting the next record.",
    "query": "Which value is active for {key}? Reply with the value only, or unknown.",
}
BLOCKS = {
    "long_session": 16,
    "distant_revoke": 12,
    "mixed_lifecycle": 12,
    "duplicate_noop": 12,
}
BOUNDARIES = {
    "long_session": (5, 7, 9, 12, 14, 16),
    "distant_revoke": (6, 8, 10, 12),
    "mixed_lifecycle": (4, 7, 9, 12),
    "duplicate_noop": (4, 6, 9, 12),
}


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def replay(events):
    records = []
    for event in events:
        if event["op"] == "reset":
            records = []
        elif event["op"] == "revoke":
            records = [row for row in records if row[0] != event["batch"]]
        elif event["op"] == "assign":
            if any(batch == event["batch"] for batch, _ in records):
                raise ValueError("duplicate record identifier")
            records.append((event["batch"], event["facts"]))
        else:
            raise ValueError(f"unknown event: {event['op']}")
    state = {}
    for _, facts in records:
        state.update(facts)
    return state


def render(events):
    lines = []
    for event in events:
        if event["op"] == "reset":
            lines.append(TEMPLATES["reset"])
        elif event["op"] == "revoke":
            lines.append(TEMPLATES["revoke"].format(batch=event["batch"]))
        else:
            lines.append(TEMPLATES["header"].format(batch=event["batch"]))
            lines.extend(
                TEMPLATES["fact"].format(key=key, value=value)
                for key, value in event["facts"].items()
            )
    return "\n".join(lines)


def _prelude(suite, block, record_ids):
    if suite == "long_session":
        operations = {
            5: ("revoke", 1),
            7: ("revoke", 2),
            9: ("revoke", 4),
            12: None,
            14: ("reset", None),
            16: ("revoke", 1),
        }
    elif suite == "distant_revoke":
        operations = {6: ("revoke", 3), 8: ("revoke", 4),
                      10: ("revoke", 5), 12: ("revoke", 6)}
    elif suite == "mixed_lifecycle":
        operations = {4: ("reset", None), 7: ("revoke", 2),
                      9: ("reset", None), 12: ("revoke", 2)}
    else:
        # Batch 2 is overwritten before block 6, then revoked again at block 12.
        # Both removals are semantic no-ops for the currently visible value.
        operations = {4: None, 6: ("revoke", 4), 9: None, 12: ("revoke_id", 2)}
    operation = operations.get(block)
    if operation is None:
        return []
    kind, lag = operation
    if kind == "reset":
        return [{"op": "reset"}]
    target_block = lag if kind == "revoke_id" else block - lag
    return [{"op": "revoke", "batch": record_ids[target_block - 1]}]


def generate_group(seed, suite, group_index):
    if suite not in SUITES:
        raise ValueError(f"unknown suite: {suite}")
    group_id = hashlib.sha256(f"{seed}/{suite}/{group_index}".encode()).hexdigest()[:16]
    rng = random.Random(int(group_id, 16))
    block_count = BLOCKS[suite]
    anchor = "asset_" + group_id[:8]
    keys = ["unit_" + group_id[8:] + f"_{index}" for index in range(6)]
    record_ids = [
        "record_" + hashlib.sha256(f"{group_id}/{index}".encode()).hexdigest()[:10]
        for index in range(block_count)
    ]
    values = [f"code_{group_id}_" + value for value in rng.sample(
        [f"{index:05d}" for index in range(10000, 99999)], 80
    )]
    output = []
    for twin in (0, 1):
        blocks, events, questions, answers, semantic = [], [], [], [], []
        previous_assignment = None
        for block in range(1, block_count + 1):
            current = _prelude(suite, block, record_ids)
            key = keys[(block - 1) % len(keys)]
            value = values[2 + block]
            if suite == "duplicate_noop" and block in (4, 9):
                key, value = previous_assignment
            if suite == "duplicate_noop" and block == 5:
                # Overwrite batch 2's key so its removal at block 6 is a no-op.
                key = keys[1]
            sensor_block = 2 if suite == "duplicate_noop" and block == 5 else block
            facts = {
                key: value,
                f"sensor_{group_id}_{sensor_block}": values[30 + block],
            }
            if block == 1:
                facts[anchor] = values[twin]
            if suite in ("long_session", "mixed_lifecycle") and block in (4, 14, 9):
                if current and current[0]["op"] == "reset":
                    facts[anchor] = values[20]
            assignment = {"op": "assign", "batch": record_ids[block - 1], "facts": facts}
            current.append(assignment)
            previous_assignment = (key, value)
            events.extend(current)
            semantic.append(copy.deepcopy(current))
            blocks.append({"index": block, "text": render(current)})
            if block in BOUNDARIES[suite]:
                historical_key = keys[(block - 3) % len(keys)]
                for role, query_key in (
                    ("current", key),
                    ("historical", historical_key),
                    ("anchor", anchor),
                ):
                    identifier = f"b{block}_{role}"
                    questions.append(
                        {
                            "id": identifier,
                            "after_block": block,
                            "text": TEMPLATES["query"].format(key=query_key),
                        }
                    )
                    answers.append(
                        {
                            "id": identifier,
                            "answer": replay(events).get(query_key, UNKNOWN),
                            "role": role,
                            "key": query_key,
                        }
                    )
        identifier = f"{group_id}.{suite}.{twin}"
        inp = {
            "protocol": PROTOCOL,
            "id": identifier,
            "group_id": group_id,
            "split": suite,
            "rules": RULE,
            "blocks": blocks,
            "decision_boundaries": list(BOUNDARIES[suite]),
            "queries": questions,
        }
        target = {
            "id": identifier,
            "answers": answers,
            "family": suite,
            "decisions": len(BOUNDARIES[suite]),
            "twin": twin,
            "semantic_events": semantic,
            "token_weight_oracle_available": False,
        }
        output.append((inp, target))
    return output


def validate(pairs):
    if not pairs:
        raise ValueError("empty corpus")
    index = {inp["id"]: (inp, target) for inp, target in pairs}
    if len(index) != len(pairs):
        raise ValueError("duplicate episode")
    group_suites = {}
    symbols = {suite: set() for suite in SUITES}
    counts, group_counts = Counter(), Counter()
    for inp, target in pairs:
        suite, group_id = inp["split"], inp["group_id"]
        if suite not in SUITES or inp["protocol"] != PROTOCOL or target["id"] != inp["id"]:
            raise ValueError("identity mismatch")
        if group_id in group_suites and group_suites[group_id] != suite:
            raise ValueError("group appears in multiple suites")
        group_suites[group_id] = suite
        if len(inp["blocks"]) != BLOCKS[suite]:
            raise ValueError("block count mismatch")
        if len(target["semantic_events"]) != BLOCKS[suite]:
            raise ValueError("semantic event count mismatch")
        if inp["decision_boundaries"] != list(BOUNDARIES[suite]):
            raise ValueError("boundary mismatch")
        if len(inp["queries"]) != 3 * len(BOUNDARIES[suite]):
            raise ValueError("query count mismatch")
        if target["token_weight_oracle_available"]:
            raise ValueError("weight oracle leaked")
        answer_map = {row["id"]: row for row in target["answers"]}
        all_events = []
        for block, events in zip(inp["blocks"], target["semantic_events"]):
            if block["text"] != render(events):
                raise ValueError("render mismatch")
            all_events.extend(events)
            for event in events:
                if event["op"] == "assign":
                    symbols[suite].update(event["facts"])
                    symbols[suite].update(event["facts"].values())
            truth = replay(all_events)
            for query in inp["queries"]:
                if query["after_block"] != block["index"]:
                    continue
                answer = answer_map[query["id"]]
                if answer["answer"] != truth.get(answer["key"], UNKNOWN):
                    raise ValueError("causal answer mismatch")
                if query["text"] != TEMPLATES["query"].format(key=answer["key"]):
                    raise ValueError("query render mismatch")
                if answer["answer"] != UNKNOWN and answer["answer"] in query["text"]:
                    raise ValueError("query leaks answer")
        twin_id = inp["id"].rsplit(".", 1)[0] + "." + str(1 - target["twin"])
        other_inp, other_target = index[twin_id]
        if other_inp["queries"] != inp["queries"]:
            raise ValueError("twin query mismatch")
        left = target["semantic_events"][0][0]["facts"]
        right = other_target["semantic_events"][0][0]["facts"]
        if set(left) != set(right) or sum(left[key] != right[key] for key in left) != 1:
            raise ValueError("twin must change one initial fact")
        if target["semantic_events"][1:] != other_target["semantic_events"][1:]:
            raise ValueError("twin changed later events")
        counts[suite] += 1
        group_counts[group_id] += 1
    if set(group_counts.values()) != {2}:
        raise ValueError("every group must contain exactly two twins")
    for left in SUITES:
        for right in SUITES:
            if left != right and symbols[left] & symbols[right]:
                raise ValueError("identifier or value leakage across suites")
    return {
        "episodes": len(pairs),
        "groups": len(group_suites),
        "suite_episodes": dict(counts),
        "queries": sum(len(inp["queries"]) for inp, _ in pairs),
        "causal_answers_verified": True,
        "suite_identifiers_and_values_disjoint": True,
        "complete_twins": True,
        "oracle_actions_generated": False,
    }


def materialize(seed, groups_per_suite, output):
    if groups_per_suite < 1:
        raise ValueError("groups_per_suite must be positive")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    pairs = [
        pair
        for suite in SUITES
        for group in range(groups_per_suite)
        for pair in generate_group(seed, suite, group)
    ]
    audit = validate(pairs)
    output.mkdir(parents=True)
    manifest = {
        "protocol": PROTOCOL,
        "seed": seed,
        "groups_per_suite": groups_per_suite,
        "suites": list(SUITES),
        "blocks": BLOCKS,
        "boundaries": {key: list(value) for key, value in BOUNDARIES.items()},
        "audit": audit,
        "files_sha256": {},
        "templates_sha256": digest(TEMPLATES),
        "training_status": "frozen_controller_evaluation_only",
        "oracle_status": "not_generated_by_builder",
    }
    for suite in SUITES:
        selected = [(inp, target) for inp, target in pairs if inp["split"] == suite]
        for suffix, rows in (
            ("inputs", [inp for inp, _ in selected]),
            ("targets", [target for _, target in selected]),
        ):
            path = output / f"{suite}.{suffix}.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            manifest["files_sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=202610154)
    parser.add_argument("--groups-per-suite", type=int, default=64)
    args = parser.parse_args()
    print(json.dumps(materialize(args.seed, args.groups_per_suite, args.output), indent=2))


if __name__ == "__main__":
    main()
