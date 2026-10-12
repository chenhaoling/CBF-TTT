"""Projection and multi-seed aggregation tests."""

import unittest

from tasks.train_cbf_qwen_lowrank_forgetting import SEEDS, development_gate, multi_seed_summary


class SummaryTests(unittest.TestCase):
    def test_registered_seeds_and_gate(self):
        groups = {str(i): {"accuracy": .95, "nll": .1} for i in range(4)}
        policies = {}
        for name, accuracy in (("fixed_retain", .6), ("fixed_half", .65), ("fixed_local", .75),
                               ("fixed_global", .5), ("global_learned", .78),
                               ("learned", .96), ("oracle", .97)):
            policies[name] = {"mean_accuracy": accuracy, "mean_nll": 1 - accuracy,
                              "min_seed_accuracy": accuracy,
                              "per_seed": {str(seed): {"accuracy": accuracy, "nll": 1 - accuracy,
                                                       "groups": {g: {"accuracy": accuracy, "nll": 1 - accuracy} for g in groups}}
                                           for seed in SEEDS}}
        gate = development_gate(policies)
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["stronger_control"], "global_learned")


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch tests run on experiment server")
class EmbeddingTests(unittest.TestCase):
    def test_frozen_transform_applies_to_new_names(self):
        from tasks.train_cbf_qwen_lowrank_forgetting import build_embeddings
        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                return [ord(char) % 31 for char in text]
        class Model:
            def __init__(self):
                self.embedding = torch.nn.Embedding(32, 12)
                self.lm_head = torch.nn.Linear(12, 32, bias=False)
            def get_input_embeddings(self): return self.embedding
        def pairs(entity, value):
            return [({}, {"semantic_events": [[{"op": "assign", "facts": {entity: value}}]]})]
        model = Model(); tokenizer = Tokenizer()
        train, _, transforms = build_embeddings(model, tokenizer, pairs("unit_a", "code_1") + pairs("unit_b", "code_2"), 8)
        test, _, _ = build_embeddings(model, tokenizer, pairs("unit_c", "code_3"), 8, transforms)
        for table in (train, test):
            for memory in table.values():
                self.assertTrue(all(torch.isfinite(value).all() for value in memory.values.values()))
                self.assertTrue(all(abs(float(value.norm()) - 1) < 1e-5 for value in memory.values.values()))


if __name__ == "__main__": unittest.main()
