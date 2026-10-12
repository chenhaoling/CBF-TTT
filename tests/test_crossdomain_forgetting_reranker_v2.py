"""Threshold-selection tests for cross-domain forgetting reranker V2."""

import unittest

from tasks.train_crossdomain_forgetting_reranker_v2 import choose_threshold
from scripts.audit_crossdomain_forgetting_reranker_v2 import oracle_metrics


class CrossDomainForgettingRerankerV2Tests(unittest.TestCase):
    def test_threshold_prioritizes_worst_difficult_action(self):
        rows = [
            {"target": 8, "top": 0, "probability": .6,
             "event_kind": "primary_initial"},
            {"target": 8, "top": 0, "probability": .4,
             "event_kind": "hard_retain"},
            {"target": 0, "top": 0, "probability": .6,
             "event_kind": "primary_update"},
        ]
        threshold, _ = choose_threshold(rows)
        self.assertGreater(threshold, .4)
        self.assertLessEqual(threshold, .6)

    def test_oracle_metrics_recompute_fold_specific_thresholds(self):
        rows = [
            {"target": 8, "top": 0, "probability": .4, "threshold": .5,
             "event_kind": "primary_initial"},
            {"target": 8, "top": 0, "probability": .4, "threshold": .5,
             "event_kind": "hard_retain"},
            {"target": 0, "top": 0, "probability": .6, "threshold": .5,
             "event_kind": "primary_update"},
        ]
        result = oracle_metrics(rows)
        self.assertEqual(result["candidate_top1_accuracy"], 1)
        self.assertEqual(result["action_accuracy"], 1)


if __name__ == "__main__":
    unittest.main()
