import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_cbf_update_scale import summarize


class UpdateScaleSummaryTests(unittest.TestCase):
    def test_paired_groups_and_direction_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scale.jsonl"
            rows = []
            for group in range(2):
                for regime in ("new_only", "old_only"):
                    losses = [0.9, 1.0, 1.02] if regime == "new_only" else [1.1, 1.0, 0.98]
                    rows.append({"protocol": "update_scale_diagnostic_v1",
                                 "id": f"{group}-{regime}", "group_id": str(group),
                                 "regime": regime, "scales": [-1.0, 0.0, 1.0],
                                 "losses": losses, "candidate_ratio": 0.1,
                                 "chunk_surprise": 2.0, "time_s": 1.0,
                                 "peak_reserved_gib": 4.0})
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            result = summarize(path)
            self.assertEqual(result["source_groups"], 2)
            self.assertEqual(result["by_regime"]["new_only"]["negative_scale_benefit_groups"], 2)
            self.assertEqual(result["by_regime"]["old_only"]["positive_scale_benefit_groups"], 2)
            rows.pop()
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            with self.assertRaisesRegex(ValueError, "each source group"):
                summarize(path)


if __name__ == "__main__":
    unittest.main()
