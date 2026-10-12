"""Tests for the frozen-Qwen ZsRE scope resolver protocol."""

import unittest

from tasks.train_zsre_scope_resolver import (
    ARMS, HEAD_SEEDS, candidate_counts, development_gate, feature_tensor,
    query_groups, split_calibration,
)


try:
    import torch
except ImportError:
    torch = None


def session(index, facts=2):
    return {"id": f"session_{index}", "facts": [
        {"subject": f"subject_{index}_{j}", "prompt": f"prompt {index} {j}",
         "rephrase_prompt": f"rephrase {index} {j}",
         "locality_prompts": [f"local a {index} {j}", f"local b {index} {j}"]}
        for j in range(facts)]}


class ProtocolTests(unittest.TestCase):
    def test_registered_arms_and_seeds(self):
        self.assertEqual(ARMS, ("bi_cosine", "cross_linear"))
        self.assertEqual(HEAD_SEEDS, (7101, 7102, 7103))

    def test_calibration_split_is_deterministic_and_disjoint(self):
        sessions = [session(index) for index in range(34)]
        train, tune = split_calibration(sessions)
        train2, tune2 = split_calibration(list(reversed(sessions)))
        self.assertEqual([row["id"] for row in train], [row["id"] for row in train2])
        self.assertEqual([row["id"] for row in tune], [row["id"] for row in tune2])
        self.assertEqual((len(train), len(tune)), (27, 7))
        self.assertFalse({row["id"] for row in train} & {row["id"] for row in tune})

    def test_query_targets_include_retrieve_and_defer(self):
        groups = query_groups([session(0)], "calibration")
        self.assertEqual(len(groups), 8)
        self.assertEqual([row["type"] for row in groups].count("locality"), 4)
        self.assertEqual([row["target"] for row in groups], [0, 0, 8, 8, 1, 1, 8, 8])

    def test_gate_requires_all_metrics_and_control(self):
        primary = {"by_type": {kind: {"accuracy": .95} for kind in
                               ("exact", "rephrase", "locality")},
                   "min_seed_accuracy": .9}
        comparison = {"overall": {"accuracy_gain": .08, "bootstrap_95ci": [.03, .12]}}
        self.assertTrue(development_gate(primary, comparison)["passed"])
        primary["by_type"]["locality"]["accuracy"] = .89
        self.assertFalse(development_gate(primary, comparison)["passed"])


@unittest.skipIf(torch is None, "torch tests run on experiment server")
class HeadTests(unittest.TestCase):
    def test_scope_head_has_finite_nonzero_gradient(self):
        from tasks.train_zsre_scope_resolver import ScopeHead
        head = ScopeHead(5, 3); values = torch.randn(6, 3, 5); targets = torch.tensor([0, 1, 2, 0, 1, 2])
        loss = torch.nn.functional.cross_entropy(head.logits(values), targets); loss.backward()
        gradient = sum(float(parameter.grad.abs().sum()) for parameter in head.parameters())
        self.assertGreater(gradient, 0.); self.assertTrue(torch.isfinite(loss))

    def test_short_session_is_padded_and_masked(self):
        from tasks.train_zsre_scope_resolver import PAIR_TEMPLATE, ScopeHead, pair_texts
        groups = query_groups([session(9, facts=3)], "calibration")
        features = {name: torch.randn(5) for name in pair_texts(groups)}
        values = feature_tensor(groups, features)
        self.assertEqual(tuple(values.shape), (12, 8, 5))
        self.assertTrue(torch.equal(candidate_counts(groups), torch.full((12,), 3)))
        logits = ScopeHead(5, 2).logits(values, candidate_counts(groups))
        self.assertTrue(torch.isneginf(logits[:, 3:8]).all())
        self.assertTrue(torch.isfinite(logits[:, [0, 1, 2, 8]]).all())


if __name__ == "__main__": unittest.main()
