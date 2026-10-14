import math
import unittest

from tasks.train_qwen_cohort_forgetting import (
    FEATURE_NAMES, GRID, apply_exact_reference, apply_retention, batch_delta,
    counterfactual_label, features, physics_retention,
)


class QwenCohortForgettingTest(unittest.TestCase):
    def state(self):
        return [
            {"batch": "old", "facts": {"a": "1"}, "delta": 1., "weight": 1.},
            {"batch": "last2", "facts": {"b": "2"}, "delta": 2., "weight": 1.},
            {"batch": "last1", "facts": {"c": "3"}, "delta": 3., "weight": 1.},
        ]

    def assignment(self, old_refs=(), prelude=0):
        return {"batch": "new", "facts": {"d": "4"},
                "old_refs": list(old_refs), "prelude_lines": prelude}

    def test_grid_is_retain_heavy_and_complete(self):
        self.assertEqual(len(GRID), 27)
        self.assertEqual(GRID[0], (1., 1., 1.))
        self.assertEqual(GRID[-1], (0., 0., 0.))

    def test_retention_targets_three_recency_cohorts(self):
        output = apply_retention(
            self.state(), (.25, .5, .75), self.assignment(), 4.)
        self.assertEqual([row["weight"] for row in output], [.25, .5, .75, 1.])

    def test_physics_maps_visible_revoke_and_reset(self):
        state = self.state()
        self.assertEqual(physics_retention(state, self.assignment(("last1",))),
                         (1., 1., 0.))
        self.assertEqual(physics_retention(state, self.assignment(("last2",))),
                         (1., 0., 1.))
        self.assertEqual(physics_retention(state, self.assignment(("old",))),
                         (0., 1., 1.))
        self.assertEqual(physics_retention(state, self.assignment((), 1)),
                         (0., 0., 0.))

    def test_exact_reference_removes_only_named_batch(self):
        output = apply_exact_reference(
            self.state(), self.assignment(("last2",)), 4.)
        self.assertEqual([row["batch"] for row in output],
                         ["old", "last1", "new"])

    def test_tensor_features_and_counterfactual_label_are_finite(self):
        try:
            import torch
            from tasks.train_cbf_lowrank_forgetting import Embeddings
        except ImportError:
            self.skipTest("torch is not installed locally")
        embeddings = Embeddings(7, 8)
        assignment = self.assignment(("last1",))
        delta = batch_delta(assignment, 4, embeddings)
        state = [{"batch": "last1", "facts": {"a": "1"},
                  "delta": torch.eye(8), "weight": 1.}]
        values = features(state, assignment, delta, 4)
        self.assertEqual(len(values), len(FEATURE_NAMES))
        self.assertTrue(all(math.isfinite(value) for value in values))
        truth = [{"key": "d", "answer": "4", "role": "current"}]
        label, outcomes, _ = counterfactual_label(
            state, assignment, delta, truth, ["1", "4", "unknown"], embeddings)
        self.assertIn(label, GRID)
        self.assertEqual(len(outcomes), 27)


if __name__ == "__main__":
    unittest.main()
