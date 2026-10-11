import unittest

from tasks.train_cbf_retrieval_head import gate, head_schedule, make_head


def rows():
    return [{"context_id": context, "split": "train"}
            for context in ("a", "b") for _ in range(4)]


class RetrievalHeadScheduleTests(unittest.TestCase):
    def test_cycles_balance_assignments(self):
        schedule = head_schedule(rows(), 3, 17)
        self.assertEqual(len(schedule), 12)
        for context in ("a", "b"):
            shifts = [shift for item, left, right in schedule if item == context
                      for shift in (left, right)]
            self.assertEqual(sorted(shifts), [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])


class RetrievalHeadModuleTests(unittest.TestCase):
    def test_head_trains_query_and_value_projections(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        torch.manual_seed(1)
        head = make_head(8, 4)
        query = torch.randn(2, 8)
        values = torch.randn(4, 8)
        logits = head(query, values)
        self.assertEqual(tuple(logits.shape), (2, 4))
        logits.sum().backward()
        self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                            for parameter in head.parameters()))


class RetrievalHeadGateTests(unittest.TestCase):
    def test_gate_requires_memory_specific_retrieval(self):
        comparison = {"counterfactual": {"nll_gain_bootstrap_95ci": [.1, .2]}}
        dev = {
            "sets": {
                "context4": {
                    "policies": {"correct": {"top1": .75}, "counterfactual": {"top1": .25}},
                    "correct_comparisons": comparison,
                },
                "global16": {"policies": {"correct": {"top1": .4}}},
            },
            "full_kv": {"top1": .6},
        }
        self.assertTrue(gate(dev)["passed"])
        dev["sets"]["context4"]["policies"]["counterfactual"]["top1"] = .6
        self.assertFalse(gate(dev)["passed"])


if __name__ == "__main__":
    unittest.main()
