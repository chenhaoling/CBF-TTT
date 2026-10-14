import unittest

from tasks.diagnose_natural_semantic_embedding_geometry import surface_folds


class NaturalSemanticEmbeddingGeometryTest(unittest.TestCase):
    def test_surface_fold_ownership(self):
        rows = [
            {"outer_fold": 0, "left_surface_id": "a", "right_surface_id": "b"},
            {"outer_fold": 0, "left_surface_id": "b", "right_surface_id": "c"},
            {"outer_fold": 1, "left_surface_id": "d", "right_surface_id": "e"},
        ]
        self.assertEqual(surface_folds(rows), {"a": 0, "b": 0, "c": 0,
                                              "d": 1, "e": 1})

    def test_cross_fold_surface_is_rejected(self):
        rows = [
            {"outer_fold": 0, "left_surface_id": "a", "right_surface_id": "b"},
            {"outer_fold": 1, "left_surface_id": "a", "right_surface_id": "c"},
        ]
        with self.assertRaises(ValueError):
            surface_folds(rows)


if __name__ == "__main__":
    unittest.main()
