"""Build the frozen smoke/pilot inputs for native fast-weight admission V1.

The semantic source stays in a separate audit file.  The model-input file has
token ids and opaque hashes only, so it cannot accidentally become an online
plaintext memory used by a controller.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


PROTOCOL = "native_support_aware_fastweight_admission_v1"
PLAN = "SUPPORT_AWARE_WRITE_GATE_V1_PLAN.md"
CHUNK_SIZE = 4096
BOUNDARIES = 3
GROUPS_PER_MECHANISM = 8
MECHANISM_SOURCE = {
    "novel_supporting": "same_subject_different_relation",
    "useful_repeat": "semantic_duplicate",
    "exact_duplicate": "explicit_reset_revoke",
    "irrelevant_distractor": "different_subject_same_relation",
    "conflicting_overwrite": "semantic_overwrite",
    "stale_repetition": "stale_repetition",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(*parts: object) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8") as sink:
        for row in rows:
            sink.write(json.dumps(row, sort_keys=True) + "\n")


def _encode(tokenizer, text: str) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        raise ValueError("text encoded to no tokens")
    return ids


def pack_chunk(tokenizer, statements: list[str], chunk_size: int = CHUNK_SIZE) -> list[int]:
    """Repeat a deterministic observation packet to one native update chunk."""
    if not statements:
        raise ValueError("a chunk needs at least one statement")
    blocks = [
        _encode(tokenizer, f"Session observation {index + 1}: {text}\n")
        for index, text in enumerate(statements)
    ]
    prefix = _encode(tokenizer, "Session observations follow.\n")
    values = list(prefix)
    cursor = 0
    while len(values) < chunk_size:
        block = blocks[cursor % len(blocks)]
        values.extend(block[: chunk_size - len(values)])
        cursor += 1
    if len(values) != chunk_size:
        raise AssertionError("chunk packing failed")
    return values


def _answer_ids(tokenizer, answer: str) -> list[int]:
    # Leading whitespace matches the answer position after the fixed prompt.
    return _encode(tokenizer, " " + answer.strip())


def _query_row(tokenizer, query: dict, group_id: str, boundary: int, index: int) -> tuple[dict, dict]:
    prompt = query["input"]["query"].rstrip() + "\nAnswer:"
    candidates = list(query["candidates"])
    if query["answer"] not in candidates or len(candidates) != 6:
        raise ValueError("query candidate contract changed")
    model = {
        "query_id": stable_hash(group_id, boundary, index, prompt)[:24],
        "role": "related" if query["role"].startswith("current_") else "protected",
        "prompt_ids": _encode(tokenizer, prompt),
        "candidate_ids": [_answer_ids(tokenizer, value) for value in candidates],
        "label": candidates.index(query["answer"]),
    }
    audit = {
        "query_id": model["query_id"],
        "role": query["role"],
        "prompt": prompt,
        "answer": query["answer"],
        "candidates": candidates,
    }
    return model, audit


def _exact_duplicate_boundary(episode: dict, boundary_index: int) -> tuple[dict, list[dict]]:
    """Turn a disjoint lifecycle source world into a literal duplicate world."""
    initial = episode["events"][0]["record"]
    source_boundary = episode["boundaries"][boundary_index]
    first_boundary = episode["boundaries"][0]
    incoming = {
        "record_id": f"{episode['source_group']}-exact-{boundary_index}",
        "input": dict(initial["input"]),
        "audit": {**initial["audit"], "role": "exact_duplicate"},
    }
    queries = []
    for query in source_boundary["queries"]:
        if query["role"].startswith("current_"):
            row = json.loads(json.dumps(query))
            row["answer"] = initial["input"]["value"]
            if row["answer"] not in row["candidates"]:
                row["candidates"][-1] = row["answer"]
            queries.append(row)
        else:
            template = next(q for q in first_boundary["queries"] if q["role"] == query["role"])
            queries.append(json.loads(json.dumps(template)))
    return incoming, queries


def convert_episode(tokenizer, episode: dict, mechanism: str) -> tuple[dict, dict]:
    group_id = f"sawg-v1-{mechanism}-{episode['source_group']}"
    setup_records = [event["record"] for event in episode["events"][:5]]
    setup_text = [record["input"]["statement"] for record in setup_records]
    model_boundaries, audit_boundaries = [], []
    for boundary_index, boundary in enumerate(episode["boundaries"]):
        if mechanism == "exact_duplicate":
            incoming, queries = _exact_duplicate_boundary(episode, boundary_index)
        else:
            incoming = episode["events"][boundary["event_index"]]["record"]
            queries = boundary["queries"]
        model_queries, audit_queries = [], []
        for index, query in enumerate(queries):
            model_query, audit_query = _query_row(tokenizer, query, group_id, boundary_index, index)
            model_queries.append(model_query); audit_queries.append(audit_query)
        model_boundaries.append({
            "boundary": boundary_index,
            "chunk_ids": pack_chunk(tokenizer, [incoming["input"]["statement"]]),
            "queries": model_queries,
        })
        audit_boundaries.append({
            "boundary": boundary_index,
            "incoming_record_id": incoming["record_id"],
            "incoming_statement": incoming["input"]["statement"],
            "incoming_role": incoming["audit"]["role"],
            "queries": audit_queries,
        })
    model = {
        "protocol": PROTOCOL,
        "group_id": group_id,
        "mechanism_hash": stable_hash("mechanism", mechanism)[:24],
        "setup_chunk_ids": pack_chunk(tokenizer, setup_text),
        "boundaries": model_boundaries,
    }
    audit = {
        "protocol": PROTOCOL,
        "group_id": group_id,
        "source_group": episode["source_group"],
        "source_episode": episode["episode_id"],
        "mechanism": mechanism,
        "source_mechanism": episode["mechanism"],
        "setup_statements": setup_text,
        "boundaries": audit_boundaries,
    }
    return model, audit


def build(source_episodes: Path, tokenizer_path: str, output: Path, plan_path: Path) -> dict:
    from transformers import AutoTokenizer

    if output.exists():
        raise FileExistsError(output)
    if not plan_path.is_file():
        raise FileNotFoundError(plan_path)
    source_audit_path = source_episodes.parent / "data_audit.json"
    source_manifest_path = source_episodes.parent / "manifest.json"
    if not source_audit_path.is_file() or not source_manifest_path.is_file():
        raise FileNotFoundError("source episodes require sibling data_audit.json and manifest.json")
    source_audit = json.loads(source_audit_path.read_text())
    if not source_audit.get("passed") or source_audit.get("episodes_sha256") != sha256(source_episodes):
        raise ValueError("source semantic corpus did not pass its independent audit")
    if source_audit.get("manifest_sha256") != sha256(source_manifest_path):
        raise ValueError("source semantic manifest changed after audit")
    episodes = read_jsonl(source_episodes)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    selected, audits = [], []
    for mechanism, source_mechanism in MECHANISM_SOURCE.items():
        candidates = [row for row in episodes
                      if row["mechanism"] == source_mechanism and row["twin"] == 0]
        if len(candidates) != GROUPS_PER_MECHANISM:
            raise ValueError(f"{source_mechanism} expected 8 source groups, got {len(candidates)}")
        for episode in sorted(candidates, key=lambda row: row["source_group"]):
            model, audit = convert_episode(tokenizer, episode, mechanism)
            selected.append(model); audits.append(audit)
    if len({row["group_id"] for row in selected}) != 48:
        raise ValueError("pilot source groups are not unique")
    output.mkdir(parents=True)
    model_path = output / "model_inputs.jsonl"
    semantic_path = output / "semantic_audit.jsonl"
    write_jsonl(model_path, selected); write_jsonl(semantic_path, audits)
    manifest = {
        "protocol": PROTOCOL,
        "stage": "pilot",
        "plan_sha256": sha256(plan_path),
        "source_episodes_sha256": sha256(source_episodes),
        "source_audit_sha256": sha256(source_audit_path),
        "source_manifest_sha256": sha256(source_manifest_path),
        "source_audit_passed": True,
        "tokenizer": tokenizer_path,
        "tokenizer_class": type(tokenizer).__name__,
        "chunk_size": CHUNK_SIZE,
        "groups": len(selected),
        "boundaries": len(selected) * BOUNDARIES,
        "smoke_groups": [row["group_id"] for mechanism in MECHANISM_SOURCE
                         for row in [x for x in selected if x["mechanism_hash"] == stable_hash("mechanism", mechanism)[:24]][:2]],
        "mechanisms": dict(Counter(row["mechanism"] for row in audits)),
        "mechanism_hashes": {name: stable_hash("mechanism", name)[:24]
                             for name in MECHANISM_SOURCE},
        "files_sha256": {
            model_path.name: sha256(model_path),
            semantic_path.name: sha256(semantic_path),
        },
        "controller_inputs_contain_semantic_fields": False,
        "formal_data_built": False,
        "test_opened": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-episodes", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=Path(PLAN))
    args = parser.parse_args()
    print(json.dumps(build(args.source_episodes, args.tokenizer, args.output, args.plan), indent=2))


if __name__ == "__main__":
    main()
