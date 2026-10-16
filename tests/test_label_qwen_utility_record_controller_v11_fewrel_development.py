"""Focused tests for V11 label scale and resource gate."""

import unittest

from tasks.label_qwen_utility_record_controller_v11_fewrel_development import (
    EXPECTED_EPISODES, EXPECTED_GROUPS, EXPECTED_LABELS, formal_gate,
)


class V11LabelTest(unittest.TestCase):
    def test_registered_scale(self):
        self.assertEqual((EXPECTED_GROUPS, EXPECTED_EPISODES, EXPECTED_LABELS),
                         (112, 224, 672))

    def test_gate_rejects_missing_benefit_support(self):
        gate = formal_gate(
            finite_complete=True, peak_reserved_gib=10,
            answer_coverage=1, positive_fraction=.049,
            oracle_headroom=.1, relation_coverage=True,
            zero_oracle_collateral=True, heldouts_sealed=True,
        )
        self.assertFalse(gate["passed"])


if __name__ == "__main__":
    unittest.main()
