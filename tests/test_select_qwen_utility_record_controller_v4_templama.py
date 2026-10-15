import unittest

from tasks.select_qwen_utility_record_controller_v4_templama import (
    CALIBRATION_GROUPS,
    LOCALITY_RULES,
    locality_allows,
    select_config,
    validation_group_split,
)


class TestSelectUtilityRecordControllerV4TempLama(unittest.TestCase):
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

    def test_selection_rejects_collateral(self):
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

    def test_validation_split_is_exact_stratified_and_deterministic(self):
        quotas = {
            "P102": 31, "P108": 52, "P127": 17, "P286": 29,
            "P39": 30, "P488": 25, "P54": 36, "P6": 23, "P69": 13,
        }
        episodes = [
            {
                "source_group": f"g-{relation}-{index}",
                "audit": {"target_relation": relation},
            }
            for relation, count in quotas.items()
            for index in range(count)
            for _ in range(2)
        ]
        left = validation_group_split(episodes)
        right = validation_group_split(list(reversed(episodes)))
        self.assertEqual(left, right)
        calibration, evaluation, allocation = left
        self.assertEqual(len(calibration), CALIBRATION_GROUPS)
        self.assertEqual(len(evaluation), CALIBRATION_GROUPS)
        self.assertFalse(calibration & evaluation)
        self.assertEqual(set(allocation), set(quotas))
        self.assertTrue(all(
            abs(row["calibration"] - row["evaluation"]) <= 1
            for row in allocation.values()
        ))


if __name__ == "__main__":
    unittest.main()
