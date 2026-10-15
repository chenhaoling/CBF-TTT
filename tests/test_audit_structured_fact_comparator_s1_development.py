import unittest

from scripts.audit_structured_fact_comparator_s1_development import release_gate


class StructuredFactComparatorS1DevelopmentAuditTests(unittest.TestCase):
    def test_independent_gate_rejects_missing_recall_gain(self):
        selected = {"exact_four_field_accuracy": .9,
                    "same_relation": {"recall": .9, "specificity": .98},
                    "delete": {"precision": 1., "recall": .4},
                    "same_entity_other_relation_specificity": 1.,
                    "relation_macro_balanced_accuracy": .9}
        baselines = {"base": {"delete": {"precision": 1., "recall": .35}}}
        self.assertFalse(release_gate(selected, 1., baselines)["passed"])


if __name__ == "__main__":
    unittest.main()
