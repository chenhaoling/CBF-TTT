import unittest

from tasks.train_qwen_natural_scope_residual_gate_v4 import ResidualGate


class NaturalScopeResidualGateTest(unittest.TestCase):
    def test_zero_initialized_residual_matches_frozen_logit(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is exercised in the remote training environment")
        base = {"weight": torch.tensor([[1., -2.]]), "bias": torch.tensor([.25])}
        features = torch.tensor([[3., 4.], [-1., 2.]])
        model = ResidualGate(2, 4, base, torch.device("cpu"))
        expected = (features @ base["weight"].t() + base["bias"]).squeeze(-1)
        torch.testing.assert_close(model(features), expected)


if __name__ == "__main__":
    unittest.main()
