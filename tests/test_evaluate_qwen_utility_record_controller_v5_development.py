import unittest

from tasks.evaluate_qwen_utility_record_controller_v5_development import (
    random_residual_matched_actions,
    safe_single_residual_oracle_actions,
)
from tasks.train_qwen_utility_record_controller_v5_residual import (
    residual_inference_candidates,
)
from tests.test_train_qwen_utility_record_controller_v5_residual import fixture


class TestEvaluateUtilityRecordControllerV5Development(unittest.TestCase):
    def test_inference_candidates_exclude_counterfactual_targets(self):
        label, feature = fixture()
        rows = residual_inference_candidates([label], [feature])
        self.assertEqual(len(rows), 2)
        self.assertTrue(all("benefit_target" not in row for row in rows))
        self.assertTrue(all("risk_target" not in row for row in rows))
        self.assertEqual(
            [row["candidate_action"] for row in rows],
            ["grid_0_1", "grid_1_0"],
        )

    def test_safe_single_oracle_selects_helpful_safe_deletion(self):
        label, feature = fixture()
        actions = safe_single_residual_oracle_actions([label], [feature])
        self.assertEqual(actions[("e", 0)], "grid_0_1")

    def test_random_control_preserves_one_residual_deletion(self):
        label, feature = fixture()
        actions = random_residual_matched_actions(
            [label], [feature], {("e", 0): "grid_0_1"}, 0
        )
        selected = label["actions"][actions[("e", 0)]]["spec"]["alphas"]
        self.assertEqual(selected.count(0.0), 1)


if __name__ == "__main__":
    unittest.main()
