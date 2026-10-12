import unittest

from tasks.train_cbf_lowrank_forgetting import development_gate


class LowRankForgettingTests(unittest.TestCase):
    def test_development_gate_requires_all_projection_seeds(self):
        def row(acc):
            groups = {"a": {"accuracy": acc, "nll": 1. - acc},
                      "b": {"accuracy": acc, "nll": 1. - acc}}
            return {"mean_accuracy": acc, "mean_nll": .1, "min_seed_accuracy": acc,
                    "per_seed": {str(seed): {"accuracy": acc, "nll": .1, "groups": groups}
                                 for seed in (1701, 1702, 1703)}}
        summary = {name: row(.7) for name in
                   ("fixed_retain", "fixed_half", "fixed_local", "fixed_global", "global_learned")}
        summary["learned"] = row(.98); summary["oracle"] = row(1.)
        self.assertTrue(development_gate(summary)["passed"])
        summary["learned"]["min_seed_accuracy"] = .8
        self.assertFalse(development_gate(summary)["passed"])

    def test_fast_weight_actions_and_counterfactual_labels(self):
        try:
            import torch
            from tasks.build_cbf_event_curriculum import generate_group
            from tasks.train_cbf_lowrank_forgetting import (
                ACTIONS, Embeddings, apply_action, batch_delta, build_examples, matrix,
            )
            from tasks.train_cbf_structured_forgetting import parse_block
        except ImportError:
            self.skipTest("torch unavailable")
        pairs = generate_group(19, "train", 0)
        embeddings = Embeddings(1701, 8)
        inp, _ = pairs[0]
        assignment = parse_block(inp["blocks"][0]["text"], [])
        delta = batch_delta(assignment, 1, embeddings)
        state = apply_action([], "retain", assignment, delta)
        half = apply_action(state, "half", assignment, delta)
        self.assertTrue(torch.allclose(matrix(half[:-1], 8), .5 * matrix(state, 8)))
        examples, meta = build_examples(pairs, embeddings)
        self.assertEqual(sum(meta["label_counts"].values()), len(examples))
        self.assertTrue(set(meta["label_counts"]).issubset(ACTIONS))


if __name__ == "__main__":
    unittest.main()
