import unittest

from tasks.extract_qwen_utility_record_controller_v1_features import (
    BASE_FEATURE_NAMES,
    FEATURE_NAMES,
    safe_binary_action,
    subject_masked_text,
    teacher_targets,
    value_masked_text,
)


def action(name, alphas, correct, unrelated, nll=1.0, mass=1.0):
    return {
        "spec": {"name": name, "kind": "grid", "alphas": alphas},
        "correct": correct,
        "unrelated_correct": unrelated,
        "total_answer_nll": nll,
        "mean_memory_mass": mass,
    }


class TestUtilityRecordControllerFeatures(unittest.TestCase):
    def test_safe_binary_teacher_enforces_barrier(self):
        label = {
            "candidate_count": 2,
            "retain_action": "grid_1_1",
            "actions": {
                "grid_0_0": action("grid_0_0", [0.0, 0.0], 7, 10),
                "grid_0_1": action("grid_0_1", [0.0, 1.0], 10, 9),
                "grid_1_0": action("grid_1_0", [1.0, 0.0], 9, 10),
                "grid_1_1": action("grid_1_1", [1.0, 1.0], 8, 10),
                "global_clear": {
                    "spec": {"name": "global_clear", "kind": "global_clear", "alphas": []},
                    "correct": 11, "unrelated_correct": 8,
                    "total_answer_nll": .5, "mean_memory_mass": 0,
                },
            },
        }
        self.assertEqual(safe_binary_action(label), "grid_1_0")
        targets = teacher_targets(label)
        self.assertEqual(targets["keep_targets"], [1, 0])
        self.assertEqual(targets["delete_targets"], [0, 1])
        self.assertEqual(targets["risk_targets"], [1, 0])

    def test_fixed_feature_schema_and_masks(self):
        record = {
            "input": {
                "statement": "Alice was born in Paris.",
                "subject": "Alice",
                "value": "Paris",
            }
        }
        self.assertEqual(len(BASE_FEATURE_NAMES), 22)
        self.assertEqual(len(FEATURE_NAMES), 66)
        self.assertNotIn("Alice", subject_masked_text(record))
        self.assertNotIn("Paris", value_masked_text(record))


if __name__ == "__main__":
    unittest.main()
