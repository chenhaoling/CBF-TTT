"""Tests for ZsRE contrastive key alignment."""

import unittest

from tasks.train_zsre_contrastive_keys import ARMS


try:
    import torch
except ImportError:
    torch = None


class ProtocolTests(unittest.TestCase):
    def test_registered_arms(self):
        self.assertEqual(ARMS, ("identity", "positive_only", "relation_aware"))


@unittest.skipIf(torch is None, "torch tests run on experiment server")
class AlignmentTests(unittest.TestCase):
    def test_zero_initialized_head_is_numerical_identity_after_normalization(self):
        from tasks.train_zsre_contrastive_keys import AlignmentHead
        torch.manual_seed(1); values = torch.nn.functional.normalize(torch.randn(7, 16), dim=-1)
        head = AlignmentHead(16, 4, 2)
        self.assertTrue(torch.allclose(head(values), values, rtol=1e-6, atol=1e-7))

    def test_relation_margin_loss_has_gradient(self):
        from tasks.train_zsre_contrastive_keys import AlignmentHead
        torch.manual_seed(3); head = AlignmentHead(8, 2, 4)
        prompt = torch.nn.functional.normalize(torch.randn(5, 8), dim=-1)
        positive = torch.nn.functional.normalize(torch.randn(5, 8), dim=-1)
        negative = torch.nn.functional.normalize(torch.randn(5, 8), dim=-1)
        p, r, n = head(prompt), head(positive), head(negative)
        loss = torch.nn.functional.softplus(.2 + (p*n).sum(-1) - (p*r).sum(-1)).mean()
        loss.backward()
        self.assertGreater(sum(float(parameter.grad.abs().sum()) for parameter in head.parameters()), 0.)


if __name__ == "__main__": unittest.main()
