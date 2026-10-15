"""Tests for Factor Gate V2 cross-fit reconstruction."""

import unittest

from tasks.build_qwen_natural_forgetting_crossfit_v4 import training_config


class CrossfitV4Tests(unittest.TestCase):
    def test_training_config_excludes_decision_thresholds(self):
        selected = {"config": {"hidden": 16, "learning_rate": .005,
                               "weight_decay": .01, "ignored": 3},
                    "benefit_threshold": .8, "risk_threshold": .1}
        self.assertEqual(training_config(selected), {
            "hidden": 16, "learning_rate": .005, "weight_decay": .01,
        })


if __name__ == "__main__":
    unittest.main()
