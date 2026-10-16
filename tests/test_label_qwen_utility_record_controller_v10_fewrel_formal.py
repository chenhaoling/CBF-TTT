import unittest

from tasks.label_qwen_utility_record_controller_v10_fewrel_formal import formal_gate


class V10FewRelFormalLabelsTest(unittest.TestCase):
    def test_formal_gate_requires_scale_consistency_and_relation_coverage(self):
        values = dict(
            finite_complete=True, peak_reserved_gib=12.0,
            answer_coverage=1.0, positive_fraction=.15,
            oracle_headroom=.10, relation_coverage=True,
            zero_oracle_collateral=True, heldouts_sealed=True,
        )
        self.assertTrue(formal_gate(**values)["passed"])
        values["relation_coverage"] = False
        self.assertFalse(formal_gate(**values)["passed"])


if __name__ == "__main__":
    unittest.main()
