import unittest

from tasks.pilot_qwen_utility_record_controller_v2_templama import (
    half_strength_decision,
)


class TestPilotUtilityRecordControllerV2TempLama(unittest.TestCase):
    def test_half_strength_requires_frequency_and_prediction_change(self):
        base = {
            "safe_oracle_contains_half": False,
            "half_vs_endpoint_correct_gain": 0,
            "half_vs_endpoint_nll_gain": 0.0,
        }
        rows = [dict(base) for _ in range(20)]
        rows[0] = {
            "safe_oracle_contains_half": True,
            "half_vs_endpoint_correct_gain": 1,
            "half_vs_endpoint_nll_gain": 0.0,
        }
        decision = half_strength_decision(rows)
        self.assertTrue(decision["requires_separate_soft_v2"])
        rows[0]["half_vs_endpoint_correct_gain"] = 0
        rows[0]["half_vs_endpoint_nll_gain"] = 0.02
        decision = half_strength_decision(rows)
        self.assertFalse(decision["requires_separate_soft_v2"])


if __name__ == "__main__":
    unittest.main()
