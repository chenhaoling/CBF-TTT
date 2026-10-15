import unittest

from scripts.inspect_lama_pararel_utility_source import normalize
from tasks.build_qwen_utility_semantic_record_gate_v1 import (
    LENGTHS,
    MECHANISMS,
    build_episode,
)
from tasks.build_qwen_utility_semantic_record_gate_v1_formal import (
    FORMAL_GROUPS,
    GROUP_PREFIX,
    choose_formal_material,
    input_source_rows,
    material_rows,
)


def fact(relation, index):
    return {
        "relation_id": relation,
        "subject": f"Subject {relation} {index}",
        "subject_norm": normalize(f"Subject {relation} {index}"),
        "value": f"Value {relation} {index}",
        "value_norm": normalize(f"Value {relation} {index}"),
        "source_fact_hash": f"fact-{relation}-{index:03d}",
    }


def source():
    relations = [f"P{index}" for index in range(8)]
    facts = {relation: [fact(relation, index) for index in range(80)]
             for relation in relations}
    patterns = {
        relation: [
            "[X] links to [Y] .", "[Y] is linked from [X] .",
            "For [X], the value is [Y] .", "[X] has [Y] .",
        ]
        for relation in relations
    }
    return relations, facts, patterns


class TestUtilitySemanticRecordFormalData(unittest.TestCase):
    def test_allocator_is_group_disjoint(self):
        relations, facts, patterns = source()
        hashes, subjects = {"reserved"}, {"reserved subject"}
        left = choose_formal_material(
            0, relations, facts, patterns, hashes, subjects
        )
        left_hashes = {row["source_fact_hash"] for row in input_source_rows(left)}
        left_subjects = {row["subject_norm"] for row in input_source_rows(left)}
        right = choose_formal_material(
            1, relations, facts, patterns, hashes, subjects
        )
        right_hashes = {row["source_fact_hash"] for row in input_source_rows(right)}
        right_subjects = {row["subject_norm"] for row in input_source_rows(right)}
        self.assertEqual(len(material_rows(left)), 12)
        self.assertEqual(len(left_hashes), 7)
        self.assertEqual(len(left_subjects), 7)
        self.assertFalse(left_hashes & right_hashes)
        self.assertFalse(left_subjects & right_subjects)

    def test_formal_identity_and_balancing(self):
        relations, facts, patterns = source()
        material = choose_formal_material(
            0, relations, facts, patterns, set(), set()
        )
        episode = build_episode(
            0, MECHANISMS[0], LENGTHS[0], 0, material,
            group_prefix=GROUP_PREFIX,
        )
        self.assertEqual(episode["source_group"], "usr-v1-formal-g000")
        mechanisms = [MECHANISMS[index % len(MECHANISMS)]
                      for index in range(FORMAL_GROUPS)]
        lengths = [LENGTHS[index % len(LENGTHS)] for index in range(FORMAL_GROUPS)]
        self.assertLessEqual(
            max(mechanisms.count(value) for value in MECHANISMS)
            - min(mechanisms.count(value) for value in MECHANISMS), 1
        )
        self.assertEqual({lengths.count(value) for value in LENGTHS}, {128})


if __name__ == "__main__":
    unittest.main()
