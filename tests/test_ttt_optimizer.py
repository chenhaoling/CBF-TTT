"""Validate CE token alignment, gradient assembly, and isolated interventions."""
import unittest
import torch
import torch.nn.functional as F
from scripts.diagnose_ttt_optimizer import standard_head_gradient, intervention_weights, displacement


class OptimizerAuditTests(unittest.TestCase):
    def test_blocked_ce_matches_full_ce_and_gradient(self):
        torch.manual_seed(108)
        h = torch.randn(2, 9, 8, requires_grad=True)
        w = torch.randn(32, 8)
        labels = torch.randint(0, 32, (2, 9))
        labels[0, 3:5] = -100
        shifted = F.pad(labels, (0, 1), value=-100)[:, 1:]
        loss = F.cross_entropy(F.linear(h, w).reshape(-1, 32), shifted.reshape(-1))
        loss.backward()
        for size in (1, 4, 128):
            value, grad = standard_head_gradient(h, w, labels, size)
            self.assertAlmostEqual(value, float(loss.detach()), places=5)
            torch.testing.assert_close(grad, h.grad, rtol=1e-5, atol=1e-6)
        with self.assertRaises(ValueError):
            standard_head_gradient(h, w, torch.full_like(labels, -100))

    def test_interventions_only_change_requested_delta(self):
        old = {"a": torch.tensor([2.]), "b": torch.tensor([4.])}
        new = {"a": torch.tensor([5.]), "b": torch.tensor([6.])}
        expected = {"full": (5., 6.), "without_last_conv": (2., 6.),
                    "only_last_conv": (5., 4.), "half_last_conv": (3.5, 6.)}
        for name, pair in expected.items():
            values = intervention_weights(old, new, name, "a")
            self.assertEqual(tuple(float(values[k]) for k in ("a", "b")), pair)
        self.assertEqual(float(old["a"]), 2.)
        self.assertEqual(float(new["a"]), 5.)
        self.assertEqual(displacement(old, new)["a"]["relative_update"], 1.5)


if __name__ == "__main__":
    unittest.main()
