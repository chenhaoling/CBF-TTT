import json
from pathlib import Path
import zipfile
import unittest

from scripts.inspect_lama_pararel_utility_source import (
    load_patterns,
    load_trex_facts,
    normalize,
    split_relations,
)


def test_normalize_and_split_are_deterministic():
    assert normalize("  New   York. ") == "new york"
    relations = [f"P{index}" for index in range(10)]
    first = split_relations(relations)
    second = split_relations(list(reversed(relations)))
    assert first == second
    assert {key: len(value) for key, value in first.items()} == {
        "train": 6, "development": 2, "confirmation": 2,
    }
    assert len({item for values in first.values() for item in values}) == 10


def test_pattern_and_trex_filters(tmp_path: Path):
    pattern_root = tmp_path / "patterns"
    pattern_root.mkdir()
    rows = [
        {"pattern": "[X] lives in [Y] ."},
        {"pattern": "[X] lives in [Y] ."},
        {"pattern": "missing slot"},
    ]
    (pattern_root / "P1.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n"
    )
    patterns, rejected = load_patterns(pattern_root)
    assert patterns == {"P1": ["[X] lives in [Y] ."]}
    assert rejected == {"invalid_slots": 1}

    archive = tmp_path / "data.zip"
    valid = {
        "uuid": "1", "predicate_id": "P1", "sub_label": "Alice",
        "obj_label": "Paris", "evidences": [],
    }
    duplicate = dict(valid, uuid="2")
    identical = dict(valid, uuid="3", sub_label="Paris")
    mismatch = dict(valid, uuid="4", predicate_id="P2")
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(
            "data/TREx/P1.jsonl",
            "\n".join(json.dumps(row) for row in (valid, duplicate, identical, mismatch)),
        )
    facts, subjects, rejected_facts = load_trex_facts(archive)
    assert facts == {"P1": {("alice", "paris")}}
    assert subjects == {"P1": {"alice"}}
    assert rejected_facts == {
        "duplicate_fact": 1,
        "identical_subject_value": 1,
        "relation_path_mismatch": 1,
    }


class TestLamaPararelSourceInspection(unittest.TestCase):
    def test_deterministic_split(self):
        test_normalize_and_split_are_deterministic()

    def test_filters(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            test_pattern_and_trex_filters(Path(directory))
