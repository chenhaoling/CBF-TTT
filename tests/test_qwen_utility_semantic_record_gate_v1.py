from tasks.build_qwen_utility_semantic_record_gate_v1 import (
    BOUNDARIES,
    LENGTHS,
    MECHANISMS,
    build_episode,
    normalize,
    semantic_key,
)


def fake_row(relation, subject, value, index):
    return {
        "relation_id": relation,
        "subject": subject,
        "subject_norm": normalize(subject),
        "value": value,
        "value_norm": normalize(value),
        "source_fact_hash": f"fact-{relation}-{index}",
    }


def material():
    relations = ["P1", "P2", "P3", "P4", "P5", "P6"]
    patterns = {
        relation: [
            "[X] links to [Y] .", "[Y] is linked from [X] .",
            "For [X], the value is [Y] .", "[X] has [Y] .",
        ]
        for relation in relations
    }
    return {
        "target": fake_row("P1", "Alice", "Old", 0),
        "versions": [fake_row("P1", f"Donor {i}", f"New {i}", i + 1) for i in range(4)],
        "same_relation_other": fake_row("P1", "Bob", "Other", 9),
        "stable": [fake_row(f"P{i+2}", f"Stable {i}", f"Value {i}", i)
                   for i in range(5)],
        "anchor_alternate": fake_row("P2", "Twin donor", "Twin value", 20),
        "patterns": patterns,
    }


def test_every_mechanism_has_exact_sparse_boundaries_and_queries():
    for mechanism in MECHANISMS:
        for length in LENGTHS:
            episode = build_episode(0, mechanism, length, 0, material())
            assert len(episode["events"]) == length
            assert len(episode["boundaries"]) == BOUNDARIES
            for boundary in episode["boundaries"]:
                assert len(boundary["queries"]) == 5
                assert all(len(row["candidates"]) == 6 for row in boundary["queries"])
                assert all(row["answer"] in row["candidates"] for row in boundary["queries"])


def test_twins_change_only_protected_anchor_value():
    left = build_episode(3, "semantic_overwrite", 16, 0, material())
    right = build_episode(3, "semantic_overwrite", 16, 1, material())
    assert left["source_group"] == right["source_group"]
    assert left["events"][0]["record"]["input"] == right["events"][0]["record"]["input"]
    assert left["events"][1]["record"]["input"]["value"] != right["events"][1]["record"]["input"]["value"]
    for index in range(2, 16):
        assert left["events"][index]["record"]["input"] == right["events"][index]["record"]["input"]
    assert semantic_key("alice", "P1") == semantic_key("alice", "P1")
    assert semantic_key("alice", "P1") != semantic_key("alice", "P2")
