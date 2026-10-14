import unittest

from tasks.extract_natural_semantic_multiview_embeddings import subject_context_indices


class NaturalSemanticMultiviewEmbeddingsTest(unittest.TestCase):
    def test_subject_context_excludes_marker_and_respects_window(self):
        text = "aa [SUBJECT] bb cc"
        offsets = [(0, 2), (3, 4), (4, 11), (11, 12), (13, 15), (16, 18)]
        self.assertEqual(subject_context_indices(offsets, text, window=1), [0, 4])

    def test_missing_or_duplicate_marker_is_rejected(self):
        with self.assertRaises(ValueError):
            subject_context_indices([(0, 3)], "foo")
        with self.assertRaises(ValueError):
            subject_context_indices([(0, 9)], "[SUBJECT] [SUBJECT]")


if __name__ == "__main__":
    unittest.main()
