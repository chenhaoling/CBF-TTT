import unittest

try:
    import torch
except ImportError:  # The repository's macOS control environment is CPU-light.
    torch = None

from tasks.train_qwen_natural_forgetting_semantic_fact_key_v9 import (
    FEATURE_NAMES,
    SEMANTIC_DIM,
    pair_features,
    semantic_scalars,
    span_compatibility,
)


class SemanticFactKeyV9Test(unittest.TestCase):
    def test_span_compatibility_separates_entity_identity(self):
        same = span_compatibility(("ada", "lovelace"), ("ada", "lovelace"))
        other = span_compatibility(("ada", "lovelace"), ("alan", "turing"))
        self.assertEqual(same[3], 1.0)
        self.assertEqual(other[1], 0.0)

    @unittest.skipIf(torch is None, "torch is unavailable")
    def test_semantic_scalars_are_symmetric(self):
        left = torch.tensor([1.0, 0.0])
        right = torch.tensor([0.0, 1.0])
        self.assertEqual(semantic_scalars(left, right),
                         semantic_scalars(right, left))

    @unittest.skipIf(torch is None, "torch is unavailable")
    def test_registered_feature_widths(self):
        left, right = "Where was Ada born?", "What is Ada's birthplace?"
        left_view = {"span_tokens": ("ada",),
                     "relation_view": "where was <ENTITY> born"}
        right_view = {"span_tokens": ("ada",),
                      "relation_view": "what is <ENTITY> birthplace"}
        basis = torch.zeros(SEMANTIC_DIM); basis[0] = 1
        full = {left: basis, right: basis.clone()}
        relation = {left_view["relation_view"]: basis,
                    right_view["relation_view"]: basis.clone()}
        for arm, names in FEATURE_NAMES.items():
            values = pair_features(left, right, left_view, right_view,
                                   full, relation, arm)
            self.assertEqual(len(values), len(names))
            self.assertTrue(all(value == value for value in values))


if __name__ == "__main__":
    unittest.main()
