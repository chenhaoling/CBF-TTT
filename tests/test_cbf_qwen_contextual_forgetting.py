"""Compatibility tests for contextual forgetting v2."""

import unittest

from tasks.train_cbf_lowrank_forgetting import query_metrics


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch tests run on experiment server")
class UnknownCalibrationTests(unittest.TestCase):
    class Memory:
        dim = 2
        def get(self, kind, text):
            values = {
                ("entity", "missing"): torch.tensor([0., 1.]),
                ("entity", "stored"): torch.tensor([1., 0.]),
                ("value", "code_1"): torch.tensor([1., 0.]),
                ("value", "unknown"): torch.tensor([0., 1.]),
            }
            return values[kind, text]

    def test_unknown_threshold_is_explicitly_forwarded(self):
        state = [{"batch": "b", "facts": {"stored": "code_1"},
                  "delta": torch.tensor([[1., 0.], [0., 0.]]), "weight": 1.}]
        query = {"key": "missing", "answer": "unknown"}
        low = query_metrics(state, query, ["code_1", "unknown"], self.Memory(), unknown_raw=-.1)
        high = query_metrics(state, query, ["code_1", "unknown"], self.Memory(), unknown_raw=.75)
        self.assertFalse(low["correct"])
        self.assertTrue(high["correct"])


if __name__ == "__main__":
    unittest.main()
