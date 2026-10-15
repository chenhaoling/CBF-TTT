import unittest

from tasks.evaluate_qwen_utility_record_controller_v1_development import (
    development_split,
    grid_action,
    select_safe,
)


class TestEvaluateUtilityRecordControllerV1Development(unittest.TestCase):
    def test_development_split_is_deterministic_balanced_and_disjoint(self):
        groups = [f"group-{index}" for index in range(64)]
        calibration, evaluation = development_split(groups)
        reversed_calibration, reversed_evaluation = development_split(
            list(reversed(groups))
        )
        self.assertEqual(calibration, reversed_calibration)
        self.assertEqual(evaluation, reversed_evaluation)
        self.assertEqual(len(calibration), 32)
        self.assertEqual(len(evaluation), 32)
        self.assertFalse(calibration & evaluation)

    def test_grid_action_uses_binary_action_name_schema(self):
        self.assertEqual(grid_action([1, 0, 1]), "grid_1_0_1")

    def test_selection_rejects_unsafe_higher_accuracy_policy(self):
        selected = select_safe({
            "safe": {
                "accuracy": 0.7,
                "mean_answer_nll": 0.4,
                "added_unrelated_errors": 0,
            },
            "unsafe": {
                "accuracy": 0.9,
                "mean_answer_nll": 0.2,
                "added_unrelated_errors": 1,
            },
        })
        self.assertEqual(selected, "safe")


if __name__ == "__main__":
    unittest.main()
