"""Paired rollout analysis checks using grouped scenarios."""

import json
from pathlib import Path
import tempfile
import unittest

from scripts.analyze_cbf_controller_rollouts import analyze


class RolloutAnalysisTests(unittest.TestCase):
    def test_write_mode_compares_skip_and_always_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scenarios = root / "scenarios.jsonl"
            rollouts = root / "rollouts.jsonl"
            scenarios.write_text(json.dumps({"id": "s", "group_id": "g", "regime": "noise"}) + "\n")
            rows = [{"id": "s", "group_id": "g", "future_index": 0,
                     "policy": policy, "update_rule": "write", "mean_loss": loss,
                     "coefficients": [gate], "alphas": None}
                    for policy, loss, gate in (("baseline", 1.0, 1.0), ("0", 0.8, 0.0),
                                               ("controller", 0.9, 0.2))]
            rollouts.write_text("".join(json.dumps(row) + "\n" for row in rows))
            result = analyze({"controller": rollouts}, scenarios, root / "result.json", draws=100)
            self.assertEqual(result["update_rule"], "write")
            self.assertAlmostEqual(result["policies"]["controller"]["vs_gate1"]["mean_gain"], 0.1)
            self.assertAlmostEqual(result["policies"]["controller"]["vs_gate0"]["mean_gain"], -0.1)

    def test_grouped_pairing_and_fixed_baselines(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scenarios = root / "scenarios.jsonl"
            baseline = root / "baseline.jsonl"
            controller = root / "controller.jsonl"
            scenario_rows, baseline_rows, controller_rows = [], [], []
            for group in range(2):
                for variant in range(3):
                    scenario_id = f"g{group}-v{variant}"
                    scenario_rows.append({"id": scenario_id, "group_id": f"g{group}", "regime": "stable"})
                    for policy, loss in (("baseline", 2.0), ("1", 1.5)):
                        baseline_rows.append({"id": scenario_id, "group_id": f"g{group}",
                                              "future_index": 0, "policy": policy,
                                              "mean_loss": loss, "alphas": [0.0, float(policy == "1")]})
                    controller_rows.append({"id": scenario_id, "group_id": f"g{group}",
                                            "future_index": 0, "policy": "controller",
                                            "mean_loss": 1.4 if group == 0 else 1.6,
                                            "alphas": [0.0, 0.7]})
            for path, rows in ((scenarios, scenario_rows), (baseline, baseline_rows),
                               (controller, controller_rows)):
                path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            result = analyze({"full": baseline, "controller": controller}, scenarios,
                             root / "analysis.json", draws=100)
            self.assertEqual(result["source_groups"], 2)
            self.assertAlmostEqual(result["policies"]["controller"]["mean_nll"], 1.5)
            self.assertAlmostEqual(result["policies"]["controller"]["vs_alpha0"]["mean_gain"], 0.5)
            self.assertAlmostEqual(result["policies"]["controller"]["vs_alpha1"]["mean_gain"], 0.0)
            self.assertEqual(result["policies"]["controller"]["vs_alpha1"]
                             ["harmful_scenario_fraction"], 0.5)


if __name__ == "__main__":
    unittest.main()
