import unittest

from scripts.inspect_lama_pararel_utility_source import normalize
from tasks.build_qwen_utility_record_controller_v6_original_confirmation import (
    GROUP_PREFIX,
    GROUPS,
    MIN_RELATION_SUBJECTS,
)
from tasks.build_qwen_utility_semantic_record_gate_v1 import build_episode
from tasks.build_qwen_utility_semantic_record_gate_v1_formal import (
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


class TestV6OriginalSourceConfirmation(unittest.TestCase):
    def test_seven_relation_material_and_identity(self):
        relations = [f"P{index}" for index in range(7)]
        facts = {
            relation: [fact(relation, index) for index in range(100)]
            for relation in relations
        }
        patterns = {
            relation: [
                "[X] links to [Y] .",
                "[Y] is linked from [X] .",
                "For [X], [Y] applies .",
                "[X] has [Y] .",
            ]
            for relation in relations
        }
        material = choose_formal_material(
            0, relations, facts, patterns, set(), set(),
            strict_same_relation_value=False,
        )
        self.assertEqual(len(material_rows(material)), 12)
        self.assertEqual(len(input_source_rows(material)), 7)
        episode = build_episode(
            0, "semantic_overwrite", 8, 0, material,
            group_prefix=GROUP_PREFIX,
        )
        self.assertEqual(
            episode["source_group"], "usr-v6-original-confirmation-g000"
        )
        self.assertEqual(GROUPS, 64)
        self.assertEqual(MIN_RELATION_SUBJECTS, 128)

    def test_low_value_cardinality_confirmation_allocation(self):
        relations = [f"P{index}" for index in range(7)]
        facts = {
            relation: [fact(relation, index) for index in range(100)]
            for relation in relations
        }
        for index, row in enumerate(facts["P0"]):
            row["value"] = f"Limited {index % 5}"
            row["value_norm"] = normalize(row["value"])
        patterns = {
            relation: [
                "[X] links to [Y] .", "[Y] is linked from [X] .",
                "For [X], [Y] applies .", "[X] has [Y] .",
            ]
            for relation in relations
        }
        material = choose_formal_material(
            0, relations, facts, patterns, set(), set(),
            strict_same_relation_value=False,
        )
        self.assertEqual(len(material_rows(material)), 12)


if __name__ == "__main__":
    unittest.main()
