"""CPU-only checks for forgetting-only pairing and result aggregation."""

import json
import tempfile
import unittest
from pathlib import Path

from tasks.cbf_forgetting_only_validation import paired_scenarios, summarize


def _scenario(group: str, regime: str) -> dict:
    return {
        "id": f"{group}-{regime}",
        "group_id": group,
        "split": "test",
        "regime": regime,
        "context_ids": [1, 2],
        "futures": [{"continuation_ids": [], "queries": [
            {"kind": "historical", "query_ids": [3], "answer_ids": [4]},
            {"kind": "current_or_new", "query_ids": [5], "answer_ids": [6]},
        ]}],
    }


class ForgettingOnlyValidationTests(unittest.TestCase):
    def test_pairing_is_cross_group_and_regime_matched(self):
        scenarios = [_scenario(group, regime)
                     for group in ("g0", "g1", "g2", "g3")
                     for regime in ("correction", "stable", "topic_shift")]
        pairs = paired_scenarios(list(reversed(scenarios)))
        self.assertEqual(len(pairs), 6)
        for left, right in pairs:
            self.assertNotEqual(left["group_id"], right["group_id"])
            self.assertEqual(left["regime"], right["regime"])

    def test_summary_uses_historical_query_for_memory_gate(self):
        policies = ("fixed_0", "fixed_0.5", "fixed_1",
                    "controller_seed42", "controller_seed43", "controller_seed44")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for policy_index, policy in enumerate(policies):
                path = root / f"{policy}.jsonl"
                paths.append(path)
                with path.open("w", encoding="utf-8") as sink:
                    for group_index, group in enumerate(("g0", "g1", "g2", "g3")):
                        donor = f"g{group_index ^ 1}"
                        for regime in ("correction", "stable", "topic_shift"):
                            correct_historical = 1.0
                            if policy.startswith("controller"):
                                correct_historical = 0.98
                            row = {
                                "id": f"{group}-{regime}", "group_id": group, "split": "test",
                                "regime": regime, "donor_id": f"{donor}-{regime}",
                                "donor_group_id": donor, "pair_index": group_index // 2,
                                "policy_name": policy, "policy": "controller" if policy.startswith("controller") else "1",
                                "coefficients": [1.0], "memory_norm": 1.0, "donor_memory_norm": 1.0,
                                "query_kinds": ["historical", "current_or_new"],
                                "query_losses": {
                                    "full_context": [correct_historical, 4.0],
                                    "correct_memory": [correct_historical, 9.0],
                                    "wrong_memory": [1.2, 2.0],
                                    "empty_memory": [1.3, 1.0],
                                },
                            }
                            row["mean_nll"] = {
                                key: sum(values) / len(values) for key, values in row["query_losses"].items()
                            }
                            sink.write(json.dumps(row) + "\n")
            output = root / "summary.json"
            result = summarize(paths, output, draws=200, seed=7)
            self.assertTrue(result["policies"]["fixed_1"]["memory_content_gate"])
            self.assertEqual(
                result["policies"]["fixed_1"]["correct_gain_vs_empty_memory"]["query_kind"],
                "historical",
            )
            self.assertTrue(result["adaptive_forgetting_gate"])
            self.assertTrue(output.is_file())


if __name__ == "__main__":
    unittest.main()
