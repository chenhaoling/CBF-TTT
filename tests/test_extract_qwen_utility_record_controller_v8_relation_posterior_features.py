import unittest
import importlib.util

from tasks.extract_qwen_utility_record_controller_v8_relation_posterior_features import (
    FEATURE_NAMES,
    POSTERIOR_FEATURE_NAMES,
    compatibility_features,
    relation_mentions,
)


def record(identity, statement, subject, value, relation):
    return {
        "record_id": identity,
        "input": {"statement": statement, "subject": subject, "value": value},
        "audit": {"relation_id": relation},
    }


class TestV8RelationPosteriorFeatures(unittest.TestCase):
    def test_registered_dimension(self):
        self.assertEqual(len(FEATURE_NAMES), 74)
        self.assertEqual(len(POSTERIOR_FEATURE_NAMES), 8)

    def test_ambiguous_training_text_is_excluded(self):
        same = record("a", "Alice likes Paris", "Alice", "Paris", "P1")
        conflict = record("b", "Bob likes Rome", "Bob", "Rome", "P2")
        domains = {
            "train": {"training": True, "episodes": [{"events": [
                {"record": same}, {"record": conflict},
            ]}]},
        }
        value = relation_mentions(domains)
        self.assertEqual(len(value["ambiguous_training_texts"]), 1)
        self.assertEqual(value["training_examples"], [])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch unavailable")
    def test_compatibility_statistics_and_rank(self):
        import torch
        candidate = [torch.tensor([3.0, 0.0]), torch.tensor([2.0, 0.0]),
                     torch.tensor([1.0, 0.0])]
        incoming = [torch.tensor([3.0, 0.0]), torch.tensor([2.0, 0.0]),
                    torch.tensor([1.0, 0.0])]
        values = compatibility_features(candidate, incoming, [.8, .2], 0)
        self.assertEqual(len(values), 8)
        self.assertEqual(values[4], 1.0)
        self.assertAlmostEqual(values[-1], .5)
        self.assertGreater(values[-2], 0)


if __name__ == "__main__":
    unittest.main()
