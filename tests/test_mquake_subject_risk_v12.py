import unittest

from scripts.audit_mquake_subject_risk_v12 import selected_threshold
from tasks.train_mquake_subject_risk_v12 import (
    candidate_metrics,
    cluster_metrics,
    equal_subject_weights,
)


def record(subject, pair=1, safety=1):
    decisions = [
        {"event_kind": "primary_old", "correct": safety},
        {"event_kind": "hard_retain", "correct": safety},
        {"event_kind": "distractor", "correct": safety},
        {"event_kind": "primary_update", "correct": pair},
    ]
    return {
        "episode_id": f"{subject}-{pair}-{safety}",
        "subject": subject,
        "length": 16,
        "decisions": decisions,
        "pair_state_correct": pair,
        "stale_deleted": pair,
        "current_retained": safety,
        "non_target_retention": float(safety),
        "memory_fraction": .5,
    }


class SubjectRiskV12Test(unittest.TestCase):
    def test_subject_weights_have_equal_total_mass(self):
        subjects = ["large"] * 100 + ["small"] * 2 + ["one"]
        weights = equal_subject_weights(subjects)
        totals = {
            subject: sum(weight for value, weight in zip(subjects, weights)
                         if value == subject)
            for subject in set(subjects)
        }
        self.assertAlmostEqual(totals["large"], totals["small"])
        self.assertAlmostEqual(totals["small"], totals["one"])
        self.assertAlmostEqual(sum(weights) / len(weights), 1.)

    def test_cluster_and_candidate_metrics_are_subject_macro(self):
        rows = [record("large") for _ in range(9)] + [record("small", pair=0)]
        metric = cluster_metrics(rows)
        self.assertEqual(metric["subjects"], 2)
        self.assertEqual(metric["pair_macro"], .5)
        candidates = [
            {"subject": "large", "label": 1., "candidate_correct": 1,
             "shortlist_correct": 1}
            for _ in range(9)
        ] + [{"subject": "small", "label": 1., "candidate_correct": 0,
              "shortlist_correct": 1}]
        candidate = candidate_metrics(candidates)
        self.assertEqual(candidate["candidate_top1"], .9)
        self.assertEqual(candidate["subject_macro_top1"], .5)

    def test_auditor_reconstructs_feasible_selection(self):
        def row(threshold, pair, lcb, feasible):
            return {
                "metric": {"pair_state_accuracy": pair, "stale_deletion": pair},
                "cluster": {"pair_macro": pair, "safety_lcb95": lcb,
                            "safety_macro": lcb},
                "feasible": feasible,
            }
        grid = {
            "0.1": row(.1, .95, .94, False),
            "0.3": row(.3, .91, .98, True),
            "0.5": row(.5, .88, .99, True),
        }
        self.assertEqual(selected_threshold(grid), (.3, True))


if __name__ == "__main__":
    unittest.main()
