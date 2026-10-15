"""Build the one-time preregistered causal-ledger lifecycle OOD corpus."""

from __future__ import annotations

import argparse
import copy
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from tasks.build_qwen_causal_ledger_residual import (
    OOD_SEED,
    PHASE_VALUE_OFFSETS,
    VALUE_POOL_SIZE,
    candidate_values,
    code_value,
    digest,
    null_value,
    parse_visible_block,
    render,
)
from tasks.build_qwen_dynamic_record_forgetting import replay
from tasks.pack_cbf_event_warmup import sha


PROTOCOL = "qwen_causal_ledger_ood_v1_corpus_v1"
PHASE = "ood"
GROUPS_PER_SUITE = 64
SUITES = (
    "long_context",
    "two_distant_revokes",
    "restore_older",
    "reset_then_duplicate",
    "mixed_unseen",
)
SUITE_SPEC = {
    "long_context": "length 24/32 with otherwise familiar retention",
    "two_distant_revokes": "two independently addressable old records revoked together",
    "restore_older": "newest assignment revoked so an older active value reappears",
    "reset_then_duplicate": "session reset followed by idempotent duplicate writes",
    "mixed_unseen": "unseen reset/revoke/overwrite compositions",
}


def boundaries(length):
    return (length // 2, (3 * length) // 4, length)


def _record_id(seed, group_id, index):
    return "archive_" + hashlib.sha256(
        f"{seed}/{group_id}/{index}".encode()
    ).hexdigest()[:12]


def _base_facts(keys, values, group_id, length, block):
    key = keys[(block - 1) % len(keys)]
    return key, {
        key: values[2 + block],
        f"probe_ood_{group_id}_{block}": values[length + 4 + block],
    }


def _event_for_block(
    suite, block, decision_boundaries, records, assignments, keys, values,
    group_id, length, twin, anchor, restore_key,
):
    boundary_index = (
        decision_boundaries.index(block) if block in decision_boundaries else None
    )
    prelude = []
    key, facts = _base_facts(keys, values, group_id, length, block)
    if block == 1:
        facts[anchor] = values[twin]

    if suite == "two_distant_revokes" and boundary_index is not None:
        lags = (6 + boundary_index, 7 + boundary_index)
        prelude = [
            {"op": "revoke", "record": records[block - lag - 1]}
            for lag in lags
        ]
    elif suite == "restore_older":
        # One overwrite three records before every boundary; revoking it must
        # restore the preceding active assignment of this key.
        if block in {1, *(value - 3 for value in decision_boundaries)}:
            key = restore_key
            facts[restore_key] = values[2 * length + block]
        if boundary_index is not None:
            prelude = [{"op": "revoke", "record": records[block - 4]}]
            if key == restore_key:
                key, facts = _base_facts(keys, values, group_id, length, block)
    elif suite == "reset_then_duplicate" and boundary_index is not None:
        if boundary_index == 0:
            prelude = [{"op": "reset"}]
            facts[anchor] = values[2 * length + 8]
        else:
            facts = copy.deepcopy(assignments[-1]["facts"])
            key = next(name for name in facts if name.startswith("marker_ood_"))
    elif suite == "mixed_unseen" and boundary_index is not None:
        if boundary_index == 0:
            prelude = [{"op": "reset"}]
            facts[anchor] = values[2 * length + 8]
        elif boundary_index == 1:
            prelude = [
                {"op": "revoke", "record": records[block - 4]},
                {"op": "revoke", "record": records[block - 2]},
            ]
            target = assignments[block - 5]
            overwrite_key = next(
                name for name in target["facts"] if name.startswith("marker_ood_")
            )
            key = overwrite_key
            facts = {
                key: values[2 + block],
                f"probe_ood_{group_id}_{block}": values[length + 4 + block],
            }
        else:
            prelude = [{"op": "revoke", "record": records[block - 3]}]
            facts = copy.deepcopy(assignments[-1]["facts"])
            key = next(name for name in facts if name.startswith("marker_ood_"))

    assignment = {"op": "assign", "record": records[block - 1], "facts": facts}
    return prelude + [assignment], key


def _historical_key(suite, boundary_index, block, keys, assignments, truth, current):
    if suite == "restore_older":
        return next(name for name in assignments[0]["facts"] if name.startswith("restore_"))
    if suite == "two_distant_revokes":
        target = assignments[block - (6 + boundary_index) - 1]
        return next(name for name in target["facts"] if name.startswith("marker_ood_"))
    if suite in {"reset_then_duplicate", "mixed_unseen"} and boundary_index == 0:
        # Deliberately query a pre-reset key to exercise null routing.
        return next(name for name in assignments[1]["facts"] if name.startswith("marker_ood_"))
    present = [name for name in keys if name != current and name in truth]
    return present[0] if present else next(name for name in keys if name != current)


def generate_group(seed, suite, group_index):
    if seed != OOD_SEED or suite not in SUITES:
        raise ValueError("registered OOD seed or suite changed")
    group_id = hashlib.sha256(
        f"ledger/{seed}/{PHASE}/{suite}/{group_index}".encode()
    ).hexdigest()[:16]
    rng = random.Random(int(group_id, 16))
    length = (24, 32)[group_index % 2] if suite == "long_context" else 16
    decision_boundaries = boundaries(length)
    anchor = f"anchor_ood_{group_id[:8]}"
    restore_key = f"restore_ood_{group_id[8:]}"
    keys = [f"marker_ood_{group_id[8:]}_{index}" for index in range(8)]
    records = [_record_id(seed, group_id, index) for index in range(length)]
    offset = PHASE_VALUE_OFFSETS[PHASE]
    indices = rng.sample(range(offset, offset + VALUE_POOL_SIZE - 1), 3 * length + 64)
    values = [code_value(PHASE, index) for index in indices]
    outputs = []
    for twin in (0, 1):
        blocks, events, semantic, queries, answers, assignments = [], [], [], [], [], []
        for block in range(1, length + 1):
            current, current_key = _event_for_block(
                suite, block, decision_boundaries, records, assignments, keys,
                values, group_id, length, twin, anchor, restore_key,
            )
            assignment = current[-1]
            assignments.append(copy.deepcopy(assignment))
            events.extend(copy.deepcopy(current))
            semantic.append(copy.deepcopy(current))
            blocks.append({"index": block, "text": render(current, PHASE)})
            if block not in decision_boundaries:
                continue
            boundary_index = decision_boundaries.index(block)
            truth = replay(events)
            historical = _historical_key(
                suite, boundary_index, block, keys, assignments, truth, current_key
            )
            for role, query_key in (
                ("current", current_key),
                ("historical", historical),
                ("anchor", anchor),
            ):
                query_id = f"b{block}_{role}"
                semantic_answer = truth.get(query_key)
                answer = semantic_answer if semantic_answer is not None else null_value(PHASE)
                queries.append({
                    "id": query_id,
                    "after_block": block,
                    "key": query_key,
                    "role": role,
                    "text": (
                        "Report the live token for marker " + query_key
                        + ", with no extra text."
                    ),
                    "candidates": candidate_values(
                        answer, values, PHASE, f"{group_id}/{twin}/{query_id}"
                    ),
                })
                answers.append({
                    "id": query_id,
                    "answer": answer,
                    "semantic_answer": semantic_answer,
                    "absent": semantic_answer is None,
                    "role": role,
                    "key": query_key,
                })
        episode_id = f"{group_id}.{suite}.{twin}"
        outputs.append(({
            "protocol": PROTOCOL,
            "id": episode_id,
            "group_id": group_id,
            "phase": PHASE,
            "family": suite,
            "blocks": blocks,
            "decision_boundaries": list(decision_boundaries),
            "queries": queries,
            "null_value": null_value(PHASE),
        }, {
            "id": episode_id,
            "family": suite,
            "length": length,
            "twin": twin,
            "answers": answers,
            "semantic_events": semantic,
            "oracle_actions_generated": False,
        }))
    return outputs


def validate(pairs, groups_per_suite=GROUPS_PER_SUITE):
    by_id = {inp["id"]: (inp, target) for inp, target in pairs}
    if len(by_id) != len(pairs):
        raise ValueError("duplicate OOD episode")
    counts, groups = Counter(), set()
    absent = Counter()
    for inp, target in pairs:
        suite = inp["family"]
        if (
            inp["protocol"] != PROTOCOL or inp["phase"] != PHASE
            or suite not in SUITES or target["family"] != suite
            or target["id"] != inp["id"] or target["oracle_actions_generated"]
            or len(inp["decision_boundaries"]) != 3 or len(inp["queries"]) != 9
        ):
            raise ValueError("OOD identity or sparse-query mismatch")
        groups.add(inp["group_id"]); counts[suite] += 1
        truth_rows = {row["id"]: row for row in target["answers"]}
        events = []
        for block, semantic in zip(inp["blocks"], target["semantic_events"]):
            if block["text"] != render(semantic, PHASE):
                raise ValueError("OOD rendering mismatch")
            parsed = parse_visible_block(block["text"], PHASE)
            assignment = semantic[-1]
            if parsed["record"] != assignment["record"] or parsed["facts"] != assignment["facts"]:
                raise ValueError("OOD visible assignment mismatch")
            events.extend(copy.deepcopy(semantic)); state = replay(events)
            for query in inp["queries"]:
                if query["after_block"] != block["index"]:
                    continue
                answer = truth_rows[query["id"]]
                expected = state.get(answer["key"])
                code = expected if expected is not None else inp["null_value"]
                if answer["semantic_answer"] != expected or answer["answer"] != code:
                    raise ValueError("OOD causal answer mismatch")
                if len(query["candidates"]) != 6 or query["candidates"].count(code) != 1:
                    raise ValueError("OOD candidate mismatch")
                absent[suite] += int(expected is None)
        twin_id = inp["id"].rsplit(".", 1)[0] + f".{1 - target['twin']}"
        if twin_id not in by_id:
            raise ValueError("missing OOD counterfactual twin")
        other_target = by_id[twin_id][1]
        if other_target["semantic_events"][1:] != target["semantic_events"][1:]:
            raise ValueError("OOD twin changed non-anchor events")
    expected_episodes = 2 * groups_per_suite
    if any(counts[suite] != expected_episodes for suite in SUITES):
        raise ValueError("OOD suite count mismatch")
    null_suites = ("two_distant_revokes", "reset_then_duplicate", "mixed_unseen")
    if any(absent[suite] == 0 for suite in null_suites):
        raise ValueError("OOD lifecycle suite lacks an absent-value query")
    return {
        "episodes": len(pairs),
        "groups": len(groups),
        "counts": dict(counts),
        "absent_queries": dict(absent),
        "causal_answers_verified": True,
        "visible_parser_verified": True,
        "counterfactual_twins_verified": True,
        "oracle_actions_generated": False,
    }


def materialize(output, groups_per_suite=GROUPS_PER_SUITE, seed=OOD_SEED):
    if groups_per_suite != GROUPS_PER_SUITE or seed != OOD_SEED:
        raise ValueError("registered OOD size or seed changed")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    pairs = [
        pair for suite in SUITES for group in range(groups_per_suite)
        for pair in generate_group(seed, suite, group)
    ]
    audit = validate(pairs, groups_per_suite)
    output.mkdir(parents=True)
    inputs = output / "ood.inputs.jsonl"
    targets = output / "ood.targets.jsonl"
    inputs.write_text("".join(json.dumps(inp) + "\n" for inp, _ in pairs))
    targets.write_text("".join(json.dumps(target) + "\n" for _, target in pairs))
    manifest = {
        "protocol": PROTOCOL,
        "phase": PHASE,
        "seed": seed,
        "groups_per_suite": groups_per_suite,
        "suites": list(SUITES),
        "suite_spec_sha256": digest(SUITE_SPEC),
        "value_pool": {
            "offset": PHASE_VALUE_OFFSETS[PHASE],
            "size": VALUE_POOL_SIZE,
            "null": null_value(PHASE),
        },
        "files_sha256": {path.name: sha(path) for path in (inputs, targets)},
        "audit": audit,
        "model_scored": False,
        "confirmation_generated": True,
        "ood_generated": True,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("protocol") != PROTOCOL or manifest.get("phase") != PHASE:
        raise ValueError("unexpected causal-ledger OOD corpus")
    paths = [root / f"ood.{kind}.jsonl" for kind in ("inputs", "targets")]
    for path in paths:
        if sha(path) != manifest["files_sha256"].get(path.name):
            raise ValueError("causal-ledger OOD hash mismatch")
    inputs = [json.loads(line) for line in paths[0].read_text().splitlines() if line]
    targets = [json.loads(line) for line in paths[1].read_text().splitlines() if line]
    if len(inputs) != len(targets) or any(
        inp["id"] != target["id"] for inp, target in zip(inputs, targets)
    ):
        raise ValueError("causal-ledger OOD alignment mismatch")
    return manifest, list(zip(inputs, targets))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--groups-per-suite", type=int, default=GROUPS_PER_SUITE)
    parser.add_argument("--seed", type=int, default=OOD_SEED)
    args = parser.parse_args()
    print(json.dumps(materialize(args.output, args.groups_per_suite, args.seed), indent=2))


if __name__ == "__main__":
    main()
