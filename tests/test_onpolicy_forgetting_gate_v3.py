"""Pure accounting tests for the on-policy forgetting gate."""

import unittest

from tasks.train_onpolicy_forgetting_gate_v3 import RETAIN_LOSS_WEIGHT


class OnPolicyForgettingGateV3Tests(unittest.TestCase):
    def test_irreversible_retain_cost_is_asymmetric(self):
        self.assertGreater(RETAIN_LOSS_WEIGHT, 1)


if __name__ == "__main__":
    unittest.main()
