"""Selection and gate checks for the Qwen-grounded low-rank bridge."""

import unittest

from tasks.cbf_qwen_lowrank_bridge import gate


class GateTests(unittest.TestCase):
    def table(self, correct=.9, wrong=.1, zero=.1, random=.1, nll=.5):
        comparisons = {}
        for name, accuracy in (("wrong", wrong), ("zero", zero), ("random", random)):
            comparisons[f"correct_vs_{name}_candidate_correct_gain"] = {"mean": correct - accuracy, "ci95": [correct - accuracy, correct - accuracy]}
            comparisons[f"correct_vs_{name}_nll_gain"] = {"mean": nll, "ci95": [nll / 2, nll]}
        return {"policies": {"correct": {"candidate_correct": correct, "greedy_correct": correct, "eos_first": 0.},
                             "wrong": {}, "zero": {}, "random": {}},
                "comparisons": comparisons, "domains": {"fineweb": correct, "longcrawl": correct}}

    def design(self):
        return {"gate": {"candidate_accuracy": .75, "domain_candidate_accuracy": .75,
                         "greedy_accuracy": .75, "accuracy_gain": .5,
                         "nll_gain": .2, "eos_rate_max": .05}}

    def test_gate_pass_and_content_fail(self):
        self.assertTrue(gate(self.table(), self.design()))
        self.assertFalse(gate(self.table(wrong=.8), self.design()))
        self.assertFalse(gate(self.table(nll=.1), self.design()))


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch tests run on experiment server")
class MemoryTests(unittest.TestCase):
    def test_rank_one_self_read_and_random_norm(self):
        from tasks.cbf_qwen_lowrank_bridge import random_value
        key = torch.randn(16); key = key / key.norm()
        value = torch.randn(16); value = value / value.norm()
        matrix = torch.outer(value, key)
        torch.testing.assert_close(matrix @ key, value)
        random, ratio = random_value(value, 10)
        self.assertAlmostEqual(ratio, 1., places=6)
        self.assertFalse(torch.equal(random, value))


if __name__ == "__main__": unittest.main()
