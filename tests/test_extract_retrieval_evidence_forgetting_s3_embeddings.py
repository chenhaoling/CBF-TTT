import json
import tempfile
import unittest
from pathlib import Path

from tasks.extract_retrieval_evidence_forgetting_s3_embeddings import partition


class RetrievalEvidenceForgettingS3EmbeddingTests(unittest.TestCase):
    def test_partition_is_complete_and_disjoint(self):
        rows = [{"id": index} for index in range(11)]
        left = partition(rows, 0, 2)
        right = partition(rows, 1, 2)
        self.assertEqual({index for index, _ in left} & {index for index, _ in right}, set())
        self.assertEqual(sorted(index for index, _ in left + right), list(range(11)))

    def test_partition_rejects_invalid_shard(self):
        with self.assertRaises(ValueError):
            partition([], 2, 2)


if __name__ == "__main__":
    unittest.main()
