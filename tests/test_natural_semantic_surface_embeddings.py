import unittest

from tasks.extract_natural_semantic_surface_embeddings import (
    partition_surfaces,
    select_audit_indices,
    token_ids_sha256,
)


class NaturalSemanticSurfaceEmbeddingsTest(unittest.TestCase):
    def test_partition_is_complete_disjoint_and_ordered(self):
        rows = [{"surface_id": str(index)} for index in range(17)]
        left = partition_surfaces(rows, 0, 2)
        right = partition_surfaces(rows, 1, 2)
        self.assertEqual([index for index, _ in left], list(range(0, 17, 2)))
        self.assertEqual([index for index, _ in right], list(range(1, 17, 2)))
        self.assertEqual({index for index, _ in left} | {index for index, _ in right},
                         set(range(17)))
        self.assertFalse({index for index, _ in left} & {index for index, _ in right})

    def test_invalid_partition_is_rejected(self):
        with self.assertRaises(ValueError):
            partition_surfaces([], 2, 2)
        with self.assertRaises(ValueError):
            partition_surfaces([], 0, 0)

    def test_audit_sample_spans_full_order(self):
        indices = select_audit_indices(15198, 32)
        self.assertEqual(len(indices), 32)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 15197)
        self.assertEqual(indices, sorted(set(indices)))

    def test_token_hash_is_order_sensitive(self):
        self.assertEqual(token_ids_sha256([1, 2, 3]), token_ids_sha256([1, 2, 3]))
        self.assertNotEqual(token_ids_sha256([1, 2, 3]), token_ids_sha256([3, 2, 1]))


if __name__ == "__main__":
    unittest.main()
