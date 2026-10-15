import unittest

from tasks.diagnose_qwen_utility_record_controller_v7_cross_source import (
    best_domain_candidate,
    diagnose_class,
    expected_calibration_error,
    probability_metrics,
    rank_metrics,
)


class TestV7FailureDiagnostic(unittest.TestCase):
    def test_rank_and_calibration_metrics(self):
        labels = [0, 1, 0, 1]
        scores = [.1, .9, .2, .8]
        self.assertEqual(rank_metrics(labels, scores)["auroc"], 1.0)
        value = probability_metrics(labels, scores)
        self.assertEqual(value["average_precision"], 1.0)
        self.assertAlmostEqual(value["brier"], .025)
        self.assertAlmostEqual(
            expected_calibration_error(labels, scores), .15
        )

    def test_domain_candidate_requires_zero_collateral(self):
        def row(gain, errors, precision=.5):
            return {
                "config": {
                    "epoch": 25, "utility_threshold": .5,
                    "risk_penalty": 1.0, "conformal_level": .03,
                },
                "domains": {"lama": {
                    "gain_over_lifecycle_physics": gain,
                    "added_unrelated_errors": errors,
                    "residual_deletion_precision": precision,
                    "residual_deletion_fraction": .1,
                }},
            }
        candidates = {
            "unsafe": row(.2, 1), "safe": row(.1, 0),
        }
        self.assertEqual(
            best_domain_candidate(candidates, "lama")["selected"], "safe"
        )

    def test_fixed_diagnosis_rule(self):
        def metric(gain, closure):
            return {"metric": {
                "gain_over_lifecycle_physics": gain,
                "oracle_gap_closure": closure,
                "added_unrelated_errors": 0,
            }}
        joint = {str(seed): {"lama": metric(.016, .55)}
                 for seed in (26101651, 26101652, 26101653)}
        transfer = {"lama": {
            str(seed): {"lama": metric(.03, .8)}
            for seed in (26101651, 26101652, 26101653)
        }}
        self.assertEqual(
            diagnose_class(joint, transfer)[0],
            "selection_calibration_limited",
        )
        joint = {str(seed): {"lama": metric(.008, .4)}
                 for seed in (26101651, 26101652, 26101653)}
        self.assertEqual(
            diagnose_class(joint, transfer)[0], "negative_transfer_limited"
        )
        transfer = {"lama": {
            str(seed): {"lama": metric(.01, .45)}
            for seed in (26101651, 26101652, 26101653)
        }}
        self.assertEqual(
            diagnose_class(joint, transfer)[0],
            "representation_ranking_limited",
        )


if __name__ == "__main__":
    unittest.main()
