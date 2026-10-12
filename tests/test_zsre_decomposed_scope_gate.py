"""Tests for decomposed ZsRE candidate retrieval and scope gating."""

import unittest

from tasks.train_zsre_decomposed_scope_gate import ARMS, GATE_SEEDS, development_gate


try:
    import torch
except ImportError:
    torch = None


class ProtocolTests(unittest.TestCase):
    def test_registered_arms_and_seeds(self):
        self.assertEqual(ARMS, ("ensemble_threshold", "decomposed_gate"))
        self.assertEqual(GATE_SEEDS, (7301, 7302, 7303))

    def test_gate_requires_quality_and_paired_gain(self):
        primary = {"by_type": {"exact": {"accuracy": .98}, "rephrase": {"accuracy": .95},
                               "locality": {"accuracy": .94}}, "min_seed_accuracy": .94}
        comparison = {"overall": {"accuracy_gain": .02, "bootstrap_95ci": [.005, .035]}}
        self.assertTrue(development_gate(primary, comparison)["passed"])
        comparison["overall"]["bootstrap_95ci"][0] = -.001
        self.assertFalse(development_gate(primary, comparison)["passed"])


@unittest.skipIf(torch is None, "torch tests run on experiment server")
class TensorTests(unittest.TestCase):
    def test_ensemble_geometry_handles_short_sessions(self):
        from tasks.train_cbf_qwen_lowrank_forgetting import QwenEmbeddings
        from tasks.train_zsre_decomposed_scope_gate import ensemble_geometry
        groups = [{"query": "q", "keys": ["a", "b", "c"]}]
        tables = {}
        for seed in (5101, 5102, 5103):
            values = {("entity", "q"): torch.tensor([1., 0.]),
                      ("entity", "a"): torch.tensor([1., 0.]),
                      ("entity", "b"): torch.tensor([0., 1.]),
                      ("entity", "c"): torch.tensor([-1., 0.])}
            tables[str(seed)] = QwenEmbeddings(seed, 2, values)
        scores, top, geometry = ensemble_geometry(groups, tables)
        self.assertEqual(tuple(scores.shape), (1, 8)); self.assertEqual(int(top[0]), 0)
        self.assertEqual(tuple(geometry.shape), (1, 4)); self.assertTrue(torch.isneginf(scores[0, 3:]).all())

    def test_binary_gate_gradient_and_feature_shape(self):
        from tasks.train_zsre_decomposed_scope_gate import BinaryGate, compose_features
        groups = [{"keys": ["a", "b"]} for _ in range(4)]
        pair = torch.randn(4, 8, 5); top = torch.tensor([0, 1, 0, 1]); geometry = torch.randn(4, 4)
        values, mean, std = compose_features(groups, pair, top, geometry)
        self.assertEqual(tuple(values.shape), (4, 9)); self.assertTrue(torch.isfinite(mean).all())
        head = BinaryGate(9, 1); labels = torch.tensor([1., 0., 1., 0.])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(head(values), labels); loss.backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in head.parameters()), 0.)


if __name__ == "__main__": unittest.main()
