import math
import unittest

from tasks.train_cbf_listwise_binding import context_candidates, objective, optimization_gate


class ListwiseDataTests(unittest.TestCase):
    def test_context_candidates_are_four_unique_values(self):
        rows = []
        for context in ("a", "b"):
            for index in range(4):
                rows.append({"context_id": context, "answer": f"code_{index:05d}"})
        result = context_candidates(rows)
        self.assertEqual(result["a"], [f"code_{index:05d}" for index in range(4)])
        self.assertEqual(result["a"], result["b"])

    def test_context_candidates_reject_duplicates(self):
        rows = [{"context_id": "a", "answer": "same"} for _ in range(4)]
        with self.assertRaises(ValueError):
            context_candidates(rows)


class ListwiseObjectiveTests(unittest.TestCase):
    def test_fast_memory_norm_bound_preserves_gradient(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        from cbf_ttt.event_writer import bound_memory

        class Node:
            pass

        model = Node()
        model.model = Node()
        model.model.layers = [Node()]
        model.model.layers[0].mlp = Node()
        model.model.layers[0].mlp.down_proj = Node()
        model.model.layers[0].mlp.down_proj.weight = torch.ones(2, 2)
        delta = torch.full((2, 2), 2.0, requires_grad=True)
        bounded = bound_memory(model, {0: delta}, .3)[0]
        ratio = bounded.float().norm() / model.model.layers[0].mlp.down_proj.weight.norm()
        self.assertAlmostEqual(float(ratio.detach()), .3, places=6)
        bounded.sum().backward()
        self.assertTrue(torch.isfinite(delta.grad).all())
        self.assertGreater(float(delta.grad.abs().sum()), 0)

    def test_retrieval_only_excludes_lm_gradient(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        retrieval = [torch.tensor(1.0, requires_grad=True), torch.tensor(2.0, requires_grad=True)]
        balanced = torch.tensor(10.0, requires_grad=True)
        loss, mean, weight = objective(retrieval, balanced, "retrieval_only")
        self.assertEqual(weight, 0.0)
        self.assertAlmostEqual(float(mean.detach()), 1.5)
        loss.backward()
        self.assertEqual(float(balanced.grad), 0.0)

    def test_hybrid_keeps_retrieval_primary(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        retrieval = [torch.tensor(1.0, requires_grad=True)]
        balanced = torch.tensor(2.0, requires_grad=True)
        loss, _, weight = objective(retrieval, balanced, "retrieval_plus_lm")
        self.assertEqual(weight, 0.25)
        self.assertAlmostEqual(float(loss.detach()), 1.5)


class ListwiseGateTests(unittest.TestCase):
    def test_optimization_gate_requires_fit_and_development(self):
        complete = {"last32_listwise_accuracy": .75, "last32_retrieval_loss": 1.0}
        generation = {"full_kv": {"code_correct": .9}}
        comparisons = {
            "permuted": {"digit_gain_bootstrap_95ci": [.1, .2]},
            "wrong": {"digit_gain_bootstrap_95ci": [.1, .2]},
        }
        dev = {
            "generation": generation,
            "global16": {
                "policies": {"correct": {"top1": .4}, "permuted": {"top1": .2},
                             "wrong": {"top1": .2}},
                "correct_comparisons": comparisons,
            },
        }
        train = {"global16": {"policies": {
            "correct": {"top1": .5}, "permuted": {"top1": .25},
        }}}
        self.assertTrue(optimization_gate(complete, dev, train)["passed"])
        complete["last32_retrieval_loss"] = math.log(4) + .01
        self.assertFalse(optimization_gate(complete, dev, train)["passed"])


if __name__ == "__main__":
    unittest.main()
