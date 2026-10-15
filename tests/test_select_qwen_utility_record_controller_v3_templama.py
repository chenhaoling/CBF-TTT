import unittest

from tasks.select_qwen_utility_record_controller_v3_templama import (
    LOCALITY_RULES,
    locality_allows,
    select_config,
)


class TestSelectUtilityRecordControllerV3TempLama(unittest.TestCase):
    def test_registered_locality_rules(self):
        values = [0.0] * 66
        values[0] = 1.0; values[2] = 1.0
        values[4] = .992; values[10] = .1
        self.assertTrue(locality_allows(values, "exact_subject"))
        self.assertTrue(locality_allows(values, "subject_value_conflict"))
        self.assertTrue(locality_allows(values, "conflict_pair_positive"))
        self.assertTrue(locality_allows(values, "conflict_cosine_099"))
        self.assertFalse(locality_allows(values, "conflict_cosine_0995"))
        self.assertTrue(locality_allows(values, "conflict_pair_cosine_099"))

    def test_selection_rejects_collateral_and_uses_frozen_ties(self):
        base = {
            "accuracy": .8, "deletion_precision": .9,
            "deletion_fraction": .2, "added_unrelated_errors": 0,
            "config": {
                "locality_rule": "exact_subject", "utility_threshold": .5,
                "risk_penalty": 1.0, "conformal_level": .03,
            },
        }
        candidates = {
            "unsafe": {**base, "accuracy": .9, "added_unrelated_errors": 1},
            "lower": {**base, "accuracy": .79},
            "selected": base,
        }
        self.assertEqual(select_config(candidates), "selected")
        self.assertIn("unrestricted", LOCALITY_RULES)


if __name__ == "__main__":
    unittest.main()
