"""Build fresh causal-ledger corpora for frozen-Qwen residual readout."""

from __future__ import annotations

import argparse
import copy
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re

from tasks.build_qwen_dynamic_record_forgetting import FAMILIES, replay
from tasks.build_controlled_interference_v18 import opaque_value
from tasks.pack_cbf_event_warmup import sha


PROTOCOL = "qwen_causal_ledger_residual_corpus_v1"
PILOT_SEED = 202610159
DEVELOPMENT_SEED = 202610160
CONFIRMATION_SEED = 202610161
OOD_SEED = 202610162
PHASES = ("pilot", "development", "confirmation")
PHASE_VALUE_OFFSETS = {
    "pilot": 0,
    "development": 16**4 // 4,
    "confirmation": 2 * (16**4 // 4),
    "ood": 3 * (16**4 // 4),
}
VALUE_POOL_SIZE = 16**4 // 4
TEMPLATES = {
    "pilot": {
        "header": "Create cache card {record}.",
        "fact": "Map {key} to code {value}.",
        "revoke": "Withdraw cache card {record}.",
        "reset": "Flush all earlier cache cards.",
        "query": "What active code belongs to {key}? Reply with only the code.",
    },
    "development": {
        "header": "Open memory ticket {record}.",
        "fact": "Under {key}, register code {value}.",
        "revoke": "Retire memory ticket {record}.",
        "reset": "Start a new memory epoch; expire every prior ticket.",
        "query": "Return the currently valid code for {key}, and nothing else.",
    },
    "confirmation": {
        "header": "Append ledger slip {record}.",
        "fact": "Code {value} is assigned to {key}.",
        "revoke": "Invalidate ledger slip {record}.",
        "reset": "Close the old ledger epoch and invalidate its slips.",
        "query": "Which code is now valid for {key}? Output only that code.",
    },
}


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def null_value(phase):
    return opaque_value(PHASE_VALUE_OFFSETS[phase] + VALUE_POOL_SIZE - 1)


def render(events, phase):
    template = TEMPLATES[phase]
    lines = []
    for event in events:
        if event["op"] == "reset":
            lines.append(template["reset"])
        elif event["op"] == "revoke":
            lines.append(template["revoke"].format(record=event["record"]))
        elif event["op"] == "assign":
            lines.append(template["header"].format(record=event["record"]))
            lines.extend(
                template["fact"].format(key=key, value=value)
                for key, value in event["facts"].items()
            )
        else:
            raise ValueError(f"unknown event operation: {event['op']}")
    return "\n".join(lines)


def _pattern(template, names):
    value = re.escape(template)
    for name in names:
        value = value.replace(re.escape("{" + name + "}"), f"(?P<{name}>.+?)")
    return re.compile("^" + value + "$")


def parse_visible_block(text, phase):
    template = TEMPLATES[phase]
    header = _pattern(template["header"], ("record",))
    revoke = _pattern(template["revoke"], ("record",))
    fact = _pattern(template["fact"], ("key", "value"))
    lines = text.splitlines()
    headers = [(index, header.fullmatch(line)) for index, line in enumerate(lines)]
    headers = [(index, match) for index, match in headers if match is not None]
    if len(headers) != 1:
        raise ValueError("visible block must contain one current record")
    header_index, match = headers[0]
    old_refs, reset_count, facts = [], 0, {}
    for index, line in enumerate(lines):
        if index < header_index:
            reference = revoke.fullmatch(line)
            if line == template["reset"]:
                reset_count += 1
            elif reference is not None:
                old_refs.append(reference.group("record"))
            else:
                raise ValueError("unknown ledger prelude")
        elif index > header_index:
            assignment = fact.fullmatch(line)
            if assignment is None:
                raise ValueError("unknown ledger fact")
            key, value = assignment.group("key"), assignment.group("value")
            if not key or not value or key in facts:
                raise ValueError("invalid ledger fact")
            facts[key] = value
    if not facts or reset_count > 1 or (reset_count and old_refs):
        raise ValueError("invalid ledger lifecycle block")
    if len(old_refs) > 2 or len(old_refs) != len(set(old_refs)):
        raise ValueError("invalid ledger references")
    return {
        "record": match.group("record"),
        "facts": facts,
        "old_refs": old_refs,
        "reset": bool(reset_count),
        "text": text,
    }


def boundaries(length):
    return tuple(sorted({max(4, length // 2), max(5, 3 * length // 4), length}))


def prelude(family, block, boundary_index, records):
    if family in {"retain", "exact_duplicate"}:
        return []
    if family == "recent_revoke":
        return [{"op": "revoke", "record": records[block - 2]}]
    if family == "distant_revoke":
        lag = min(block - 1, 4 + boundary_index)
        return [{"op": "revoke", "record": records[block - lag - 1]}]
    if family == "reset":
        return [{"op": "reset"}]
    if family == "overwritten_noop":
        lag = min(block - 2, 4)
        return [{"op": "revoke", "record": records[block - lag - 1]}]
    if family == "mixed_two_event":
        far = min(block - 1, 5 + boundary_index)
        return [
            {"op": "revoke", "record": records[block - far - 1]},
            {"op": "revoke", "record": records[block - 2]},
        ]
    raise ValueError(f"unknown family: {family}")


def candidate_values(answer, values, phase, query_id):
    null = null_value(phase)
    others = sorted(set(values) - {answer, null})
    rng = random.Random(int(hashlib.sha256(query_id.encode()).hexdigest(), 16))
    rng.shuffle(others)
    count = 5 if answer == null else 4
    selected = [answer] + ([] if answer == null else [null]) + others[:count]
    if len(selected) != 6 or len(set(selected)) != 6:
        raise ValueError("ledger query candidate construction failed")
    return selected


def generate_group(seed, phase, family, group_index):
    if phase not in PHASES or family not in FAMILIES:
        raise ValueError("unknown causal-ledger phase or family")
    group_id = hashlib.sha256(
        f"ledger/{seed}/{phase}/{family}/{group_index}".encode()
    ).hexdigest()[:16]
    rng = random.Random(int(group_id, 16))
    length = (8, 12, 16)[group_index % 3]
    decision_boundaries = boundaries(length)
    anchor = f"anchor_{phase}_{group_id[:8]}"
    keys = [f"key_{phase}_{group_id[8:]}_{index}" for index in range(5)]
    records = [
        "ledger_" + hashlib.sha256(
            f"{seed}/{phase}/{group_id}/{index}".encode()
        ).hexdigest()[:12]
        for index in range(length)
    ]
    offset = PHASE_VALUE_OFFSETS[phase]
    indices = rng.sample(range(offset, offset + VALUE_POOL_SIZE - 1), 3 * length + 32)
    values = [opaque_value(index) for index in indices]
    outputs = []
    for twin in (0, 1):
        blocks, all_events, semantic, questions, answers, assignments = [], [], [], [], [], []
        for block in range(1, length + 1):
            boundary_index = (
                decision_boundaries.index(block) if block in decision_boundaries else None
            )
            current = (
                prelude(family, block, boundary_index, records)
                if boundary_index is not None else []
            )
            key = keys[(block - 1) % len(keys)]
            facts = {
                key: values[2 + block],
                f"sensor_{phase}_{group_id}_{block}": values[length + 4 + block],
            }
            if block == 1:
                facts[anchor] = values[twin]
            if boundary_index is not None and family == "exact_duplicate":
                facts = copy.deepcopy(assignments[-1]["facts"])
                key = next(name for name in facts if name.startswith("key_"))
            if boundary_index is not None and family == "overwritten_noop":
                target_record = current[0]["record"]
                target = next(row for row in assignments if row["record"] == target_record)
                key = next(name for name in target["facts"] if name.startswith("key_"))
                facts = {
                    key: values[2 + block],
                    f"sensor_{phase}_{group_id}_{block}": values[length + 4 + block],
                }
            if boundary_index is not None and family == "reset":
                facts[anchor] = values[2 * length + 8 + boundary_index]
            assignment = {"op": "assign", "record": records[block - 1], "facts": facts}
            current.append(assignment); assignments.append(assignment)
            all_events.extend(copy.deepcopy(current)); semantic.append(copy.deepcopy(current))
            blocks.append({"index": block, "text": render(current, phase)})
            if boundary_index is None:
                continue
            truth = replay(all_events)
            historical_candidates = [name for name in keys if name != key and name in truth]
            historical = historical_candidates[0] if historical_candidates else keys[0]
            for role, query_key in (
                ("current", key), ("historical", historical), ("anchor", anchor)
            ):
                query_id = f"b{block}_{role}"
                semantic_answer = truth.get(query_key)
                answer = semantic_answer if semantic_answer is not None else null_value(phase)
                candidates = candidate_values(
                    answer, values, phase, f"{group_id}/{twin}/{query_id}"
                )
                questions.append({
                    "id": query_id,
                    "after_block": block,
                    "key": query_key,
                    "role": role,
                    "text": TEMPLATES[phase]["query"].format(key=query_key),
                    "candidates": candidates,
                })
                answers.append({
                    "id": query_id,
                    "answer": answer,
                    "semantic_answer": semantic_answer,
                    "absent": semantic_answer is None,
                    "role": role,
                    "key": query_key,
                })
        episode_id = f"{group_id}.{family}.{twin}"
        outputs.append(({
            "protocol": PROTOCOL,
            "id": episode_id,
            "group_id": group_id,
            "phase": phase,
            "family": family,
            "blocks": blocks,
            "decision_boundaries": list(decision_boundaries),
            "queries": questions,
            "null_value": null_value(phase),
        }, {
            "id": episode_id,
            "family": family,
            "length": length,
            "twin": twin,
            "answers": answers,
            "semantic_events": semantic,
            "oracle_actions_generated": False,
        }))
    return outputs


def validate(pairs, expected_groups=None):
    if not pairs:
        raise ValueError("empty causal-ledger corpus")
    by_id = {inp["id"]: (inp, target) for inp, target in pairs}
    if len(by_id) != len(pairs):
        raise ValueError("duplicate causal-ledger episode")
    groups, symbols, counts = {}, {phase: set() for phase in PHASES}, Counter()
    for inp, target in pairs:
        phase, family, group_id = inp["phase"], inp["family"], inp["group_id"]
        if (
            inp["protocol"] != PROTOCOL or target["id"] != inp["id"]
            or target["family"] != family or phase not in PHASES
            or family not in FAMILIES or target["oracle_actions_generated"]
        ):
            raise ValueError("causal-ledger identity mismatch")
        groups.setdefault(group_id, (phase, family))
        if groups[group_id] != (phase, family):
            raise ValueError("causal-ledger group leaked across roles")
        truth_rows = {row["id"]: row for row in target["answers"]}
        if len(inp["decision_boundaries"]) != 3 or len(inp["queries"]) != 9:
            raise ValueError("causal-ledger sparse boundary mismatch")
        events = []
        for block, semantic in zip(inp["blocks"], target["semantic_events"]):
            if block["text"] != render(semantic, phase):
                raise ValueError("causal-ledger render mismatch")
            parsed = parse_visible_block(block["text"], phase)
            assignment = semantic[-1]
            if parsed["record"] != assignment["record"] or parsed["facts"] != assignment["facts"]:
                raise ValueError("causal-ledger visible assignment mismatch")
            if parsed["old_refs"] != [
                row["record"] for row in semantic[:-1] if row["op"] == "revoke"
            ] or parsed["reset"] != any(row["op"] == "reset" for row in semantic[:-1]):
                raise ValueError("causal-ledger visible lifecycle mismatch")
            events.extend(copy.deepcopy(semantic)); state = replay(events)
            for event in semantic:
                if event["op"] == "assign":
                    symbols[phase].update(event["facts"])
                    symbols[phase].update(event["facts"].values())
            for query in inp["queries"]:
                if query["after_block"] != block["index"]:
                    continue
                answer = truth_rows[query["id"]]
                expected = state.get(answer["key"])
                if answer["semantic_answer"] != expected:
                    raise ValueError("causal-ledger answer mismatch")
                expected_code = expected if expected is not None else inp["null_value"]
                if answer["answer"] != expected_code or query["candidates"].count(expected_code) != 1:
                    raise ValueError("causal-ledger candidate truth mismatch")
                if len(query["candidates"]) != 6 or len(set(query["candidates"])) != 6:
                    raise ValueError("causal-ledger candidate cardinality mismatch")
        twin_id = inp["id"].rsplit(".", 1)[0] + f".{1 - target['twin']}"
        if twin_id not in by_id:
            raise ValueError("missing causal-ledger twin")
        other, other_target = by_id[twin_id]
        if other["queries"] == inp["queries"]:
            # Candidate ordering is twin-specific, but query text and keys must match.
            pass
        if [
            (row["after_block"], row["key"], row["role"], row["text"])
            for row in other["queries"]
        ] != [
            (row["after_block"], row["key"], row["role"], row["text"])
            for row in inp["queries"]
        ]:
            raise ValueError("causal-ledger twin changed queries")
        if other_target["semantic_events"][1:] != target["semantic_events"][1:]:
            raise ValueError("causal-ledger twin changed non-anchor events")
        left, right = target["semantic_events"][0][-1]["facts"], other_target[
            "semantic_events"
        ][0][-1]["facts"]
        if set(left) != set(right) or sum(left[key] != right[key] for key in left) != 1:
            raise ValueError("causal-ledger twin must change one anchor")
        counts[phase, family] += 1
    for left in PHASES:
        for right in PHASES:
            if left != right and symbols[left] & symbols[right]:
                raise ValueError("causal-ledger symbols leaked across phases")
    if expected_groups is not None:
        for family in FAMILIES:
            if counts["pilot", family] != 2 * expected_groups:
                raise ValueError("causal-ledger pilot family count mismatch")
    return {
        "episodes": len(pairs),
        "groups": len(groups),
        "counts": {f"{phase}/{family}": value for (phase, family), value in counts.items()},
        "causal_answers_verified": True,
        "visible_parser_verified": True,
        "counterfactual_twins_verified": True,
        "phase_symbols_disjoint": True,
        "oracle_actions_generated": False,
    }


def materialize_pilot(output, groups_per_family=8, seed=PILOT_SEED):
    if groups_per_family != 8 or seed != PILOT_SEED:
        raise ValueError("registered ledger pilot requires eight groups and its frozen seed")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    pairs = [
        pair for family in FAMILIES for group in range(groups_per_family)
        for pair in generate_group(seed, "pilot", family, group)
    ]
    audit = validate(pairs, groups_per_family)
    output.mkdir(parents=True)
    paths = {
        "inputs": output / "pilot.inputs.jsonl",
        "targets": output / "pilot.targets.jsonl",
    }
    paths["inputs"].write_text("".join(json.dumps(inp) + "\n" for inp, _ in pairs))
    paths["targets"].write_text("".join(json.dumps(target) + "\n" for _, target in pairs))
    manifest = {
        "protocol": PROTOCOL,
        "phase": "pilot",
        "seed": seed,
        "groups_per_family": groups_per_family,
        "families": list(FAMILIES),
        "templates_sha256": digest(TEMPLATES),
        "value_pool": {
            "offset": PHASE_VALUE_OFFSETS["pilot"],
            "size": VALUE_POOL_SIZE,
            "null": null_value("pilot"),
        },
        "files_sha256": {path.name: sha(path) for path in paths.values()},
        "audit": audit,
        "model_scored": False,
        "development_generated": False,
        "confirmation_generated": False,
        "ood_generated": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_pilot(root):
    root = Path(root); manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("protocol") != PROTOCOL or manifest.get("phase") != "pilot":
        raise ValueError("unexpected causal-ledger pilot protocol")
    paths = [root / f"pilot.{kind}.jsonl" for kind in ("inputs", "targets")]
    for path in paths:
        if sha(path) != manifest["files_sha256"].get(path.name):
            raise ValueError("causal-ledger pilot hash mismatch")
    inputs = [json.loads(line) for line in paths[0].read_text().splitlines() if line]
    targets = [json.loads(line) for line in paths[1].read_text().splitlines() if line]
    if len(inputs) != len(targets) or any(
        inp["id"] != target["id"] for inp, target in zip(inputs, targets)
    ):
        raise ValueError("causal-ledger pilot alignment mismatch")
    return manifest, list(zip(inputs, targets))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--groups-per-family", type=int, default=8)
    parser.add_argument("--seed", type=int, default=PILOT_SEED)
    args = parser.parse_args()
    print(json.dumps(materialize_pilot(args.output, args.groups_per_family, args.seed), indent=2))


if __name__ == "__main__":
    main()
