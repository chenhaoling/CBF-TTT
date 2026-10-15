import unittest

from tasks.evaluate_structured_fact_comparator_s1_development import gate


class StructuredFactComparatorS1DevelopmentTests(unittest.TestCase):
    def test_gate_requires_every_registered_check(self):
        selected = {
            "exact_four_field_accuracy": .9,
            "same_relation": {"recall": .9, "specificity": .98},
            "delete": {"precision": .995, "recall": .5},
            "same_entity_other_relation_specificity": .99,
            "relation_macro_balanced_accuracy": .9,
        }
        baseline = {"zero": {"delete": {"precision": 1., "recall": .3}}}
        self.assertTrue(gate(selected, .999, baseline)["passed"])
        selected["delete"]["precision"] = .98
        self.assertFalse(gate(selected, .999, baseline)["passed"])

    def test_gate_uses_best_safe_nonadapted_recall(self):
        selected = {
            "exact_four_field_accuracy": .9,
            "same_relation": {"recall": .9, "specificity": .98},
            "delete": {"precision": .995, "recall": .5},
            "same_entity_other_relation_specificity": .99,
            "relation_macro_balanced_accuracy": .9,
        }
        baseline = {"unsafe": {"delete": {"precision": .98, "recall": .9}},
                    "safe": {"delete": {"precision": .99, "recall": .35}}}
        result = gate(selected, 1., baseline)
        self.assertAlmostEqual(result["best_nonadapted_delete_recall_at_99_precision"], .35)
        self.assertTrue(result["passed"])


if __name__ == "__main__":
    unittest.main()
