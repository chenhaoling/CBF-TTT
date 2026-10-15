import unittest

from tasks.build_qwen_utility_record_controller_v2_templama_pilot import (
    PARAPHRASES,
    answer_value,
    temporal_transition,
)


class TestBuildUtilityRecordControllerV2TempLamaPilot(unittest.TestCase):
    def test_every_relation_has_three_distinct_slot_templates(self):
        self.assertEqual(len(PARAPHRASES), 9)
        for templates in PARAPHRASES.values():
            self.assertEqual(len(set(templates)), 3)
            self.assertTrue(all(value.count("[X]") == 1 and value.count("[Y]") == 1
                                for value in templates))

    def test_temporal_transition_selects_first_real_change(self):
        rows = [
            (2010, {"value_id": "Q1"}),
            (2011, {"value_id": "Q1"}),
            (2012, {"value_id": "Q2"}),
            (2013, {"value_id": "Q3"}),
        ]
        left, right = temporal_transition(rows)
        self.assertEqual(left["value_id"], "Q1")
        self.assertEqual(right["value_id"], "Q2")

    def test_answer_value_preserves_surface_and_normalizes(self):
        row = {"answer": [{"wikidata_id": "Q1", "name": "  New   York  "}]}
        self.assertEqual(answer_value(row), ("Q1", "New York", "new york"))


if __name__ == "__main__":
    unittest.main()
