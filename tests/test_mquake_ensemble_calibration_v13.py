import math
import unittest

from scripts.audit_mquake_ensemble_calibration_v13 import selected_threshold
from scripts.analyze_mquake_ensemble_calibration_v13 import (
    binary_transitions,
    decision_outcome,
)
from tasks.evaluate_mquake_ensemble_calibration_v13 import (
    EXPECTED_V12_HASHES,
    median_logit_probability,
    select_threshold_from_rows,
)


def row(threshold, pair, lcb, feasible, macro=None):
    return {
        "threshold": threshold,
        "metric": {"pair_state_accuracy": pair, "stale_deletion": pair},
        "cluster": {"pair_macro": pair if macro is None else macro,
                    "safety_lcb95": lcb, "safety_macro": lcb},
        "feasible": feasible,
    }


class EnsembleCalibrationV13Test(unittest.TestCase):
    def test_raw_median_does_not_apply_member_thresholds(self):
        self.assertAlmostEqual(median_logit_probability([.1, .2, .9]), .2)
        self.assertAlmostEqual(median_logit_probability([.01, .5, .99]), .5)
        self.assertTrue(math.isfinite(median_logit_probability([0., 1., 1.])))

    def test_feasible_selection_prioritizes_subject_macro_pair(self):
        rows = [
            row(.02, .95, .96, False, .95),
            row(.03, .90, .98, True, .88),
            row(.05, .89, .99, True, .91),
        ]
        selected, feasible = select_threshold_from_rows(rows)
        self.assertTrue(feasible)
        self.assertEqual(selected["threshold"], .05)
        grid = {str(value["threshold"]): {key: value[key]
                for key in ("metric", "cluster", "feasible")} for value in rows}
        self.assertEqual(selected_threshold(grid), (.05, True))

    def test_infeasible_selection_prioritizes_safety_lower_bound(self):
        rows = [row(.02, .95, .96, False), row(.05, .80, .99, False)]
        selected, feasible = select_threshold_from_rows(rows)
        self.assertFalse(feasible)
        self.assertEqual(selected["threshold"], .05)

    def test_locked_v12_hashes_cover_every_consumed_artifact(self):
        self.assertEqual(set(EXPECTED_V12_HASHES), {
            "summary.json", "training.json", "training_manifest.json",
            "mquake_subject_risk_crossfit.pt",
        })
        self.assertTrue(all(len(value) == 64 for value in EXPECTED_V12_HASHES.values()))

    def test_paired_diagnostic_counts_improvements_and_regressions(self):
        transition = binary_transitions([0, 0, 1, 1], [0, 1, 0, 1])
        self.assertEqual(transition["counts"], {
            "0->0": 1, "0->1": 1, "1->0": 1, "1->1": 1,
        })
        self.assertEqual(transition["delta"], 0.)
        event = {"target_session_id": "old"}
        self.assertEqual(decision_outcome(event, {
            "top": "old", "deleted": None, "probability": .2,
        })["outcome"], "correct_candidate_threshold_reject")
        self.assertEqual(decision_outcome(event, {
            "top": "old", "deleted": "old", "probability": .9,
        })["outcome"], "correct_deletion")


if __name__ == "__main__":
    unittest.main()
