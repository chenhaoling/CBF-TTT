import unittest

from tasks.train_qwen_utility_record_controller_v4_templama import (
    class_weights,
    select_checkpoint,
    stratified_group_split,
    zero_empirical_risk_threshold,
)


class TestTrainUtilityRecordControllerV4TempLama(unittest.TestCase):
    def test_group_split_is_deterministic_stratified_and_leak_free(self):
        examples = [
            {"source_group": f"g-{relation}-{group}", "relation": relation}
            for relation, count in (("P1", 11), ("P2", 16), ("P3", 24))
            for group in range(count)
            for _ in range(3)
        ]
        left = stratified_group_split(examples, 26101641)
        right = stratified_group_split(list(reversed(examples)), 26101641)
        self.assertEqual(left, right)
        train, validation, allocation = left
        self.assertFalse(train & validation)
        self.assertEqual(len(train | validation), 51)
        self.assertEqual(
            {relation: row["validation"] for relation, row in allocation.items()},
            {"P1": 2, "P2": 3, "P3": 5},
        )

    def test_group_split_rejects_cross_relation_group(self):
        with self.assertRaises(ValueError):
            stratified_group_split([
                {"source_group": "g", "relation": "P1"},
                {"source_group": "g", "relation": "P2"},
            ], 1)

    def test_checkpoint_requires_zero_unsafe_deletions(self):
        metrics = {
            25: {"unsafe_deletions": 0, "finite": True,
                 "boundary_action_accuracy": .7, "record_action_accuracy": .8,
                 "risk_recall": 1.0},
            50: {"unsafe_deletions": 1, "finite": True,
                 "boundary_action_accuracy": .9, "record_action_accuracy": .9,
                 "risk_recall": .9},
            100: {"unsafe_deletions": 0, "finite": True,
                  "boundary_action_accuracy": .7, "record_action_accuracy": .8,
                  "risk_recall": 1.0},
        }
        self.assertEqual(select_checkpoint(metrics), 25)

    def test_inverse_frequency_weights_require_both_classes(self):
        rows = [
            {"delete_target": 0, "risk_target": 0},
            {"delete_target": 0, "risk_target": 1},
            {"delete_target": 1, "risk_target": 0},
        ]
        weights = class_weights(rows)
        self.assertGreater(weights["utility"][1], weights["utility"][0])
        with self.assertRaises(ValueError):
            class_weights([{"delete_target": 0, "risk_target": 0}])

    def test_zero_empirical_risk_threshold_is_strict(self):
        threshold = zero_empirical_risk_threshold(
            [True, True, False, True], [.2, .7, .1, .5], [0, 1, 1, 1]
        )
        self.assertEqual(threshold, .5)
        predicted = [delete and risk < threshold
                     for delete, risk in zip([True, True, False, True], [.2, .7, .1, .5])]
        self.assertEqual(predicted, [True, False, False, False])


if __name__ == "__main__":
    unittest.main()
