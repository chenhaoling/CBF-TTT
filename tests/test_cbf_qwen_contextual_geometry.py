"""Tests for contextual Qwen memory geometry selection and gates."""

import unittest

from tasks.diagnose_cbf_qwen_contextual_geometry import (
    HybridEmbeddings, POOLINGS, UNKNOWN_GRID, contextual_gate, select_context_config,
)
from tasks.train_cbf_qwen_lowrank_forgetting import SEEDS


class DummyEmbeddings:
    def __init__(self, seed, dim, prefix):
        self.seed, self.dim, self.prefix = seed, dim, prefix
    def get(self, kind, text):
        return self.prefix, kind, text


class ContextualGeometryTests(unittest.TestCase):
    def test_hybrid_uses_registered_source_by_kind(self):
        key = DummyEmbeddings(1, 8, "key")
        value = DummyEmbeddings(1, 8, "value")
        hybrid = HybridEmbeddings(key, value)
        self.assertEqual(hybrid.get("entity", "x")[0], "key")
        self.assertEqual(hybrid.get("value", "y")[0], "value")

    def test_train_selection_uses_largest_near_best_regularizer(self):
        grid = {}
        accuracies = {1e-4: .95, 1e-3: .946, 1e-2: .941}
        for regularizer, best in accuracies.items():
            internal = f"context_last_r{regularizer:g}"
            grid[internal] = {}
            for threshold in UNKNOWN_GRID:
                score = best if threshold == .2 else best - .03
                grid[internal][str(threshold)] = {str(seed): score for seed in SEEDS}
        selected = select_context_config(grid, "context_last")
        self.assertEqual(selected["regularizer"], 1e-2)
        self.assertEqual(selected["unknown_raw"], .2)

    def test_gate_prefers_last_pooling(self):
        def row(value):
            return {"mean_accuracy": value, "min_seed_accuracy": value,
                    "min_family_accuracy": value}
        dev = {"context_last": row(.94), "context_mean": row(.96), "random256": row(1.)}
        sweeps = {pooling: {str(seed): {"16": {"accuracy": .9}} for seed in SEEDS}
                  for pooling in POOLINGS}
        gate = contextual_gate(dev, sweeps)
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["selected_variant"], "context_last")


if __name__ == "__main__":
    unittest.main()
