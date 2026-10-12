"""Tests for efficient action-OOF protocol constants."""

import unittest

from tasks.train_zsre_counterfactual_forgetting_controller_v3 import (
    FIXED_EPOCHS, INNER_FOLDS, TRAINING_SEED,
)
from tasks.train_zsre_forgetting_controller import CONTROLLER_SEEDS


class ActionOOFTests(unittest.TestCase):
    def test_protocol_freezes_converged_fit_once(self):
        self.assertEqual(INNER_FOLDS, 5)
        self.assertEqual(FIXED_EPOCHS, 200)
        self.assertEqual(TRAINING_SEED, CONTROLLER_SEEDS[0])


if __name__ == "__main__": unittest.main()
