"""Focused model-free tests for V11 relation-group DRO utilities."""

import unittest

from tasks.train_qwen_utility_record_controller_v11_relation_transfer import (
    ARMS, _project_group_weights, v11_gate,
)


class V11TrainingTest(unittest.TestCase):
    def test_registered_arms(self):
        self.assertEqual(ARMS, (
            "erm_equal", "relation_group_dro", "relation_group_dro_no_fewrel"))

    def test_group_weight_projection(self):
        values = _project_group_weights({"a": 100, "b": 1, "c": .0001})
        self.assertAlmostEqual(sum(values.values()), 1)
        self.assertTrue(all(value > 0 for value in values.values()))
        self.assertGreater(values["a"], values["b"])


if __name__ == "__main__":
    unittest.main()
