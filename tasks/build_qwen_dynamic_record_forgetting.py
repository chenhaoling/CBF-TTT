"""Build preregistered dynamic record-level forgetting corpora.

The builder emits semantic event streams and causal answers only.  It never
creates fast-memory actions; those are measured from the frozen Qwen geometry
by the separate resource pilot and formal trainer.
"""

from __future__ import annotations

import argparse
import copy
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re

from tasks.pack_cbf_event_warmup import sha


PROTOCOL = "qwen_dynamic_record_forgetting_corpus_v1"
DATA_SEED = 202610156
FAMILIES = (
    "retain",
    "recent_revoke",
    "distant_revoke",
    "reset",
    "exact_duplicate",
    "overwritten_noop",
    "mixed_two_event",
)
SPLITS = ("pilot", "train", "dev")
UNKNOWN = "unknown"
RULE = (
    "Records are processed in chronological order. The newest assignment in an "
    "active record wins. Deactivating a record restores the newest earlier "
    "active assignment when one exists. A session reset expires every earlier "
    "record. Repeating an assignment is idempotent. Record identifiers are "
    "references and are not values."
)
TEMPLATES = {
    "pilot": {
        "header": "Open memory record {record}:",
        "fact": "Store {key} as {value}.",
        "revoke": "Deactivate memory record {record}.",
        "reset": "Expire every earlier record in this session.",
        "query": "Return the active value for {key}, or unknown if none is active.",
    },
    "train": {
        "header": "Register state entry {record}:",
        "fact": "The binding for {key} is {value}.",
        "revoke": "State entry {record} is no longer active.",
        "reset": "Begin a fresh state; all prior entries have expired.",
        "query": "Give only the currently valid value of {key}, or unknown.",
    },
    "dev": {
        "header": "Accept ledger item {record}:",
        "fact": "Associate {value} with {key}.",
        "revoke": "Cancel ledger item {record} in full.",
        "reset": "Cross a session boundary and invalidate all older items.",
        "query": "Which value now applies to {key}? Answer only the value or unknown.",
    },
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
            records = [row for row in records if row[0] != event["record"]]
        elif event["op"] == "assign":
            if any(record == event["record"] for record, _ in records):
                raise ValueError("duplicate record identifier")
            records.append((event["record"], dict(event["facts"])))
        else:
            raise ValueError(f"unknown event operation: {event['op']}")
    state = {}
    for _, facts in records:
        state.update(facts)
    return state


def render(events, split):
    template = TEMPLATES[split]
    lines = []
    for event in events:
        if event["op"] == "reset":
            lines.append(template["reset"])
        elif event["op"] == "revoke":
            lines.append(template["revoke"].format(record=event["record"]))
        else:
            lines.append(template["header"].format(record=event["record"]))
            lines.extend(
                template["fact"].format(key=key, value=value)
                for key, value in event["facts"].items()
            )
    return "\n".join(lines)


def parse_visible_block(text, split):
    """Parse only rendered text; teacher semantic events are not consulted."""
    template = TEMPLATES[split]
    header_prefix, header_suffix = template["header"].split("{record}")
    revoke_prefix, revoke_suffix = template["revoke"].split("{record}")
    fact_pattern = re.compile(
        "^"
        + re.escape(template["fact"])
        .replace(re.escape("{key}"), "(?P<key>.+?)")
        .replace(re.escape("{value}"), "(?P<value>.+?)")
        + "$"
    )
    lines = text.splitlines()
    headers = [
        (index, line[len(header_prefix) : len(line) - len(header_suffix)])
        for index, line in enumerate(lines)
        if line.startswith(header_prefix) and line.endswith(header_suffix)
    ]
    if len(headers) != 1:
        raise ValueError("visible block must contain exactly one new record")
    header_index, record = headers[0]
    old_refs, reset_count, facts = [], 0, {}
    for index, line in enumerate(lines):
        if index < header_index:
            if line == template["reset"]:
                reset_count += 1
            elif line.startswith(revoke_prefix) and line.endswith(revoke_suffix):
                old_refs.append(
                    line[len(revoke_prefix) : len(line) - len(revoke_suffix)]
                )
            else:
                raise ValueError("unknown visible prelude")
            continue
        if index == header_index:
            continue
        match = fact_pattern.fullmatch(line)
        if match is None:
            raise ValueError("unknown visible fact line")
        key, value = match.group("key"), match.group("value")
        if not key or not value or key in facts:
            raise ValueError("invalid visible fact")
        facts[key] = value
    if not record or not facts or reset_count > 1 or (reset_count and old_refs):
        raise ValueError("invalid visible lifecycle block")
    if len(old_refs) > 2 or len(set(old_refs)) != len(old_refs):
        raise ValueError("invalid visible references")
    return {
        "record": record,
        "facts": facts,
        "old_refs": old_refs,
        "reset": bool(reset_count),
        "prelude_lines": header_index,
        "text": text,
    }


def _boundaries(length):
    return tuple(sorted({max(4, length // 2), max(5, (3 * length) // 4), length}))


def _prelude(family, block, boundary_index, record_ids):
    if family in {"retain", "exact_duplicate"}:
        return []
    if family == "recent_revoke":
        return [{"op": "revoke", "record": record_ids[block - 2]}]
    if family == "distant_revoke":
        lag = min(block - 1, 4 + boundary_index)
        return [{"op": "revoke", "record": record_ids[block - lag - 1]}]
    if family == "reset":
        return [{"op": "reset"}]
    if family == "overwritten_noop":
        # Never target block one because it owns the protected anchor.  The
        # generator reuses the target's queried key in the current assignment,
        # so removal is observationally idempotent at this boundary.
        lag = min(block - 2, 4)
        return [{"op": "revoke", "record": record_ids[block - lag - 1]}]
    if family == "mixed_two_event":
        far = min(block - 1, 5 + boundary_index)
        return [
            {"op": "revoke", "record": record_ids[block - far - 1]},
            {"op": "revoke", "record": record_ids[block - 2]},
        ]
    raise ValueError(f"unknown family: {family}")


def generate_group(seed, split, family, group_index):
    if split not in SPLITS or family not in FAMILIES:
        raise ValueError("unknown split or family")
    group_id = hashlib.sha256(
        f"{seed}/{split}/{family}/{group_index}".encode()
    ).hexdigest()[:16]
    rng = random.Random(int(group_id, 16))
    length = (8, 12, 16)[group_index % 3]
    boundaries = _boundaries(length)
    anchor = "asset_" + group_id[:8]
    keys = ["unit_" + group_id[8:] + f"_{index}" for index in range(5)]
    record_ids = [
        "record_" + hashlib.sha256(f"{group_id}/{index}".encode()).hexdigest()[:12]
        for index in range(length)
    ]
    values = [
        f"code_{group_id}_{number}"
        for number in rng.sample(range(10000, 99999), 3 * length + 32)
    ]
    outputs = []
    for twin in (0, 1):
        blocks, all_events, semantic, questions, answers = [], [], [], [], []
        assignments = []
        for block in range(1, length + 1):
            boundary_index = boundaries.index(block) if block in boundaries else None
            current = (
                _prelude(family, block, boundary_index, record_ids)
                if boundary_index is not None
                else []
            )
            key = keys[(block - 1) % len(keys)]
            facts = {
                key: values[2 + block],
                f"sensor_{group_id}_{block}": values[length + 4 + block],
            }
            if block == 1:
                facts[anchor] = values[twin]
            if boundary_index is not None and family == "exact_duplicate":
                facts = copy.deepcopy(assignments[-1]["facts"])
                key = next(name for name in facts if name.startswith("unit_"))
            if boundary_index is not None and family == "overwritten_noop":
                target_record = current[0]["record"]
                target = next(
                    row for row in assignments if row["record"] == target_record
                )
                key = next(name for name in target["facts"] if name.startswith("unit_"))
                facts = {
                    key: values[2 + block],
                    f"sensor_{group_id}_{block}": values[length + 4 + block],
                }
            if boundary_index is not None and family == "reset":
                facts[anchor] = values[2 * length + 8 + boundary_index]
            assignment = {
                "op": "assign",
                "record": record_ids[block - 1],
                "facts": facts,
            }
            current.append(assignment)
            assignments.append(assignment)
            all_events.extend(copy.deepcopy(current))
            semantic.append(copy.deepcopy(current))
            blocks.append({"index": block, "text": render(current, split)})
            if boundary_index is not None:
                truth = replay(all_events)
                historical_candidates = [
                    name for name in keys if name != key and name in truth
                ]
                historical = historical_candidates[0] if historical_candidates else keys[0]
                for role, query_key in (
                    ("current", key),
                    ("historical", historical),
                    ("anchor", anchor),
                ):
                    query_id = f"b{block}_{role}"
                    questions.append(
                        {
                            "id": query_id,
                            "after_block": block,
                            "text": TEMPLATES[split]["query"].format(key=query_key),
                        }
                    )
                    answers.append(
                        {
                            "id": query_id,
                            "answer": truth.get(query_key, UNKNOWN),
                            "role": role,
                            "key": query_key,
                        }
                    )
        episode_id = f"{group_id}.{family}.{twin}"
        outputs.append(
            (
                {
                    "protocol": PROTOCOL,
                    "id": episode_id,
                    "group_id": group_id,
                    "split": split,
                    "family": family,
                    "rules": RULE,
                    "blocks": blocks,
                    "decision_boundaries": list(boundaries),
                    "queries": questions,
                },
                {
                    "id": episode_id,
                    "family": family,
                    "length": length,
                    "twin": twin,
                    "answers": answers,
                    "semantic_events": semantic,
                    "token_weight_oracle_available": False,
                },
            )
        )
    return outputs


def validate(pairs, expected_groups_per_family=None):
    if not pairs:
        raise ValueError("empty dynamic-record corpus")
    index = {inp["id"]: (inp, target) for inp, target in pairs}
    if len(index) != len(pairs):
        raise ValueError("duplicate episode identifier")
    group_roles, symbols, counts = {}, {split: set() for split in SPLITS}, Counter()
    for inp, target in pairs:
        split, family, group_id = inp["split"], inp["family"], inp["group_id"]
        if (
            inp["protocol"] != PROTOCOL
            or target["id"] != inp["id"]
            or family != target["family"]
            or split not in SPLITS
            or family not in FAMILIES
            or target["token_weight_oracle_available"]
        ):
            raise ValueError("identity or protocol mismatch")
        role = (split, family)
        if group_id in group_roles and group_roles[group_id] != role:
            raise ValueError("source group leaked across roles")
        group_roles[group_id] = role
        if len(inp["blocks"]) != target["length"]:
            raise ValueError("block count mismatch")
        if len(target["semantic_events"]) != target["length"]:
            raise ValueError("semantic event count mismatch")
        if len(inp["decision_boundaries"]) != 3 or len(inp["queries"]) != 9:
            raise ValueError("sparse boundary/query count mismatch")
        truth_rows = {row["id"]: row for row in target["answers"]}
        if set(truth_rows) != {row["id"] for row in inp["queries"]}:
            raise ValueError("query/answer alignment mismatch")
        events = []
        for block, semantic in zip(inp["blocks"], target["semantic_events"]):
            if block["text"] != render(semantic, split):
                raise ValueError("render mismatch")
            parsed = parse_visible_block(block["text"], split)
            assignment = semantic[-1]
            if parsed["record"] != assignment["record"] or parsed["facts"] != assignment["facts"]:
                raise ValueError("visible assignment parse mismatch")
            if parsed["old_refs"] != [
                event["record"] for event in semantic[:-1] if event["op"] == "revoke"
            ]:
                raise ValueError("visible reference parse mismatch")
            if parsed["reset"] != any(event["op"] == "reset" for event in semantic[:-1]):
                raise ValueError("visible reset parse mismatch")
            before = copy.deepcopy(events)
            events.extend(copy.deepcopy(semantic))
            for event in semantic:
                if event["op"] == "assign":
                    symbols[split].update(event["facts"])
                    symbols[split].update(event["facts"].values())
            state = replay(events)
            query_keys = [
                truth_rows[query["id"]]["key"]
                for query in inp["queries"]
                if query["after_block"] == block["index"]
            ]
            if family == "overwritten_noop" and query_keys:
                without_revoke = before + [
                    copy.deepcopy(event)
                    for event in semantic
                    if event["op"] != "revoke"
                ]
                alternate = replay(without_revoke)
                if any(state.get(key, UNKNOWN) != alternate.get(key, UNKNOWN)
                       for key in query_keys):
                    raise ValueError("overwritten revoke is not a query-level no-op")
            if family == "exact_duplicate" and query_keys:
                alternate = replay(before)
                if any(state.get(key, UNKNOWN) != alternate.get(key, UNKNOWN)
                       for key in query_keys):
                    raise ValueError("duplicate assignment is not idempotent")
            for query in inp["queries"]:
                if query["after_block"] != block["index"]:
                    continue
                answer = truth_rows[query["id"]]
                if answer["answer"] != state.get(answer["key"], UNKNOWN):
                    raise ValueError("causal answer mismatch")
                if query["text"] != TEMPLATES[split]["query"].format(key=answer["key"]):
                    raise ValueError("query render mismatch")
                if answer["answer"] != UNKNOWN and answer["answer"] in query["text"]:
                    raise ValueError("answer leaked into query")
        twin_id = inp["id"].rsplit(".", 1)[0] + f".{1 - target['twin']}"
        if twin_id not in index:
            raise ValueError("missing counterfactual twin")
        other, other_target = index[twin_id]
        if other["queries"] != inp["queries"] or other_target["semantic_events"][1:] != target["semantic_events"][1:]:
            raise ValueError("counterfactual twin changed non-anchor content")
        left = target["semantic_events"][0][-1]["facts"]
        right = other_target["semantic_events"][0][-1]["facts"]
        if set(left) != set(right) or sum(left[key] != right[key] for key in left) != 1:
            raise ValueError("counterfactual twin must change one initial anchor")
        counts[split, family] += 1
    for left in SPLITS:
        for right in SPLITS:
            if left != right and symbols[left] & symbols[right]:
                raise ValueError("symbols leaked across splits")
    if expected_groups_per_family is not None:
        for family in FAMILIES:
            if counts["pilot", family] != 2 * expected_groups_per_family:
                raise ValueError("pilot family count mismatch")
    return {
        "episodes": len(pairs),
        "groups": len(group_roles),
        "counts": {f"{split}/{family}": count for (split, family), count in counts.items()},
        "causal_answers_verified": True,
        "counterfactual_twins_verified": True,
        "symbols_disjoint": True,
        "oracle_actions_generated": False,
    }


def load_corpus(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("protocol") != PROTOCOL:
        raise ValueError("dynamic-record corpus protocol mismatch")
    split = manifest["phase"]
    paths = [root / f"{split}.{kind}.jsonl" for kind in ("inputs", "targets")]
    for path in paths:
        if sha(path) != manifest["files_sha256"].get(path.name):
            raise ValueError("dynamic-record corpus hash mismatch")
    inputs = [json.loads(line) for line in paths[0].read_text().splitlines() if line]
    targets = [json.loads(line) for line in paths[1].read_text().splitlines() if line]
    if len(inputs) != len(targets) or any(
        inp["id"] != target["id"] for inp, target in zip(inputs, targets)
    ):
        raise ValueError("dynamic-record corpus alignment mismatch")
    return manifest, list(zip(inputs, targets))


def load_split(root, split):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("protocol") != PROTOCOL or split not in manifest.get("splits", []):
        raise ValueError("dynamic-record split protocol mismatch")
    paths = [root / f"{split}.{kind}.jsonl" for kind in ("inputs", "targets")]
    for path in paths:
        if sha(path) != manifest["files_sha256"].get(path.name):
            raise ValueError("dynamic-record split hash mismatch")
    inputs = [json.loads(line) for line in paths[0].read_text().splitlines() if line]
    targets = [json.loads(line) for line in paths[1].read_text().splitlines() if line]
    if len(inputs) != len(targets) or any(
        inp["id"] != target["id"] for inp, target in zip(inputs, targets)
    ):
        raise ValueError("dynamic-record split alignment mismatch")
    return list(zip(inputs, targets))


def materialize_pilot(output, groups_per_family=8, seed=DATA_SEED):
    if groups_per_family != 8:
        raise ValueError("registered pilot requires exactly eight groups per family")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    pairs = [
        pair
        for family in FAMILIES
        for group in range(groups_per_family)
        for pair in generate_group(seed, "pilot", family, group)
    ]
    audit = validate(pairs, groups_per_family)
    output.mkdir(parents=True)
    manifest = {
        "protocol": PROTOCOL,
        "phase": "pilot",
        "seed": seed,
        "groups_per_family": groups_per_family,
        "families": list(FAMILIES),
        "templates_sha256": digest(TEMPLATES),
        "audit": audit,
        "files_sha256": {},
        "training_status": "not_trained",
        "oracle_status": "not_constructed",
    }
    for kind, rows in (
        ("inputs", [inp for inp, _ in pairs]),
        ("targets", [target for _, target in pairs]),
    ):
        path = output / f"pilot.{kind}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        manifest["files_sha256"][path.name] = sha(path)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _formal_pairs(seed, split, groups):
    return [
        pair
        for ordinal in range(groups)
        for pair in generate_group(
            seed,
            split,
            FAMILIES[ordinal % len(FAMILIES)],
            ordinal // len(FAMILIES),
        )
    ]


def materialize_development(output, train_groups=512, dev_groups=64, seed=DATA_SEED):
    if (train_groups, dev_groups) != (512, 64):
        raise ValueError("registered development corpus requires 512/64 groups")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    pairs = (
        _formal_pairs(seed, "train", train_groups)
        + _formal_pairs(seed, "dev", dev_groups)
    )
    audit = validate(pairs)
    counts = Counter((inp["split"], inp["family"]) for inp, _ in pairs)
    if len({inp["group_id"] for inp, _ in pairs if inp["split"] == "train"}) != train_groups:
        raise ValueError("formal training source-group count mismatch")
    if len({inp["group_id"] for inp, _ in pairs if inp["split"] == "dev"}) != dev_groups:
        raise ValueError("formal development source-group count mismatch")
    output.mkdir(parents=True)
    manifest = {
        "protocol": PROTOCOL,
        "phase": "development",
        "seed": seed,
        "splits": ["train", "dev"],
        "group_counts": {"train": train_groups, "dev": dev_groups},
        "family_episode_counts": {
            f"{split}/{family}": counts[split, family]
            for split in ("train", "dev") for family in FAMILIES
        },
        "families": list(FAMILIES),
        "templates_sha256": digest(TEMPLATES),
        "audit": audit,
        "files_sha256": {},
        "training_status": "not_trained",
        "oracle_status": "not_constructed",
        "sealed_test_generated": False,
        "external_ood_generated": False,
    }
    for split in ("train", "dev"):
        selected = [(inp, target) for inp, target in pairs if inp["split"] == split]
        for kind, rows in (
            ("inputs", [inp for inp, _ in selected]),
            ("targets", [target for _, target in selected]),
        ):
            path = output / f"{split}.{kind}.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            manifest["files_sha256"][path.name] = sha(path)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--groups-per-family", type=int, default=8)
    parser.add_argument("--seed", type=int, default=DATA_SEED)
    parser.add_argument("--phase", choices=("pilot", "development"), default="pilot")
    args = parser.parse_args()
    manifest = (
        materialize_pilot(args.output, args.groups_per_family, args.seed)
        if args.phase == "pilot"
        else materialize_development(args.output, seed=args.seed)
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
