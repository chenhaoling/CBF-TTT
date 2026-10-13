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

    def test_pair_cache_limit_changes_eviction_not_features(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        from tasks.train_factorized_causal_forgetting_v4 import PairEvidence
        from tasks.train_long_session_sparse_gate_v1 import EMBEDDING_SEEDS

        class Values:
            def get(self, kind, value):
                del kind
                return {"x": torch.tensor([1., 0.]),
                        "y": torch.tensor([0., 1.])}[value]

        units = ["a", "b", "c", "d"]
        matrices = {str(seed): torch.eye(4) for seed in EMBEDDING_SEEDS}
        embeddings = {str(seed): Values() for seed in EMBEDDING_SEEDS}
        index = {unit: position for position, unit in enumerate(units)}
        full = {"a": "Memory statement: A\nValue: x",
                "b": "Memory statement: B\nValue: y",
                "c": "Memory statement: C\nValue: x",
                "d": "Memory statement: D\nValue: y"}
        scopes = {"a": "A", "b": "B", "c": "C", "d": "D"}
        small = PairEvidence(matrices, embeddings, index, full, scopes, cache_limit=1)
        large = PairEvidence(matrices, embeddings, index, full, scopes, cache_limit=10)
        first_small = small.pair(["a", "b"], ["c", "d"])
        first_large = large.pair(["a", "b"], ["c", "d"])
        self.assertTrue(torch.equal(first_small["scope_feature"],
                                    first_large["scope_feature"]))
        self.assertTrue(torch.equal(first_small["payload_feature"],
                                    first_large["payload_feature"]))
        small.pair(["b", "c"], ["a", "d"])
        large.pair(["b", "c"], ["a", "d"])
        self.assertEqual(small.evictions, 1)
        self.assertEqual(large.evictions, 0)


if __name__ == "__main__":
    unittest.main()
