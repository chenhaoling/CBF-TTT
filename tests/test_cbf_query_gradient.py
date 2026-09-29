"""Pure-data validation of the query-gradient diagnostic summary."""

import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_cbf_query_gradient import summarize


class QueryGradientSummaryTests(unittest.TestCase):
    def test_complete_groups_and_directional_effects(self):
        rows = []
        for group in range(3):
            for regime in ("new_only", "old_only"):
                rows.append({
                    "protocol": "query_gradient_diagnostic_v1", "id": f"{group}-{regime}",
                    "group_id": f"g{group}", "split": "train", "regime": regime,
                    "base_loss": 1.0, "gradient_loss": 1.0,
                    "candidate_norm": 1.0, "gradient_norm": 1.0,
                    "gradient_dot_candidate": 0.2, "gradient_candidate_cosine": 0.2,
                    "raw_losses": {"0.25": 1.01, "1.0": 1.02},
                    "oracle_losses": {"0.25": 0.98, "1.0": 0.99},
                    "time_s": 2.0, "peak_allocated_gib": None, "peak_reserved_gib": None,
                })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            result = summarize([path], expected_groups=3)
            self.assertEqual(result["labels"], 6)
            self.assertEqual(result["by_regime"]["new_only"]["positive_directional_derivative_groups"], 3)
            self.assertAlmostEqual(result["by_regime"]["new_only"]["mean_raw_gain_at_1"], -0.02)
            self.assertAlmostEqual(result["by_regime"]["new_only"]["mean_oracle_gain_at_0.25"], 0.02)
            self.assertTrue(result["supports_candidate_misalignment_hypothesis"])
            path.write_text("".join(json.dumps(row) + "\n" for row in rows[:-1]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "complete"):
                summarize([path], expected_groups=3)


if __name__ == "__main__":
    unittest.main()
