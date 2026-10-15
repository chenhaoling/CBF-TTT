import unittest

from tasks.train_qwen_utility_record_controller_v1 import (
    class_weights,
    relation_folds,
    select_checkpoint,
)


class TestTrainUtilityRecordControllerV1(unittest.TestCase):
    def test_relation_folds_are_deterministic_and_balanced(self):
        relations = [f"P{index}" for index in range(17)]
        left = relation_folds(relations)
        right = relation_folds(list(reversed(relations)))
        self.assertEqual(left, right)
        counts = [list(left.values()).count(index) for index in range(5)]
        self.assertLessEqual(max(counts) - min(counts), 1)

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


if __name__ == "__main__":
    unittest.main()
