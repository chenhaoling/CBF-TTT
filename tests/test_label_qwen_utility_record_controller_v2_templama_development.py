import unittest

from scripts.audit_qwen_utility_record_controller_v2_templama_development_labels import (
    expected_actions,
    independent_label,
)
from tasks.pilot_qwen_utility_semantic_record_gate_v1 import action_grid


class TestLabelUtilityRecordControllerV2TempLamaDevelopment(unittest.TestCase):
    def test_binary_action_grid_matches_independent_auditor(self):
        actions = action_grid(3, alphas=(0.0, 1.0))
        self.assertEqual({row["name"]: row for row in actions}, expected_actions(3))
        self.assertEqual(len(actions), 9)
        self.assertFalse(any(
            .5 in row["alphas"] for row in actions if row["kind"] == "grid"
        ))

    def test_independent_safe_oracle_preserves_unrelated_barrier(self):
        actions = expected_actions(1)
        row = {
            "retain_action": "grid_1",
            "actions": {
                name: {
                    "spec": spec,
                    "correct": correct,
                    "unrelated_correct": unrelated,
                    "total_answer_nll": nll,
                    "mean_memory_mass": mass,
                }
                for (name, spec), correct, unrelated, nll, mass in zip(
                    actions.items(), (8, 7, 9), (5, 4, 3), (4.0, 3.0, 2.0),
                    (0.0, 1.0, 0.0),
                )
            },
        }
        self.assertEqual(independent_label(row), "grid_0")


if __name__ == "__main__":
    unittest.main()
