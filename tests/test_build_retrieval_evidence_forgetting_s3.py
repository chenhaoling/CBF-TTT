import unittest

from tasks.build_retrieval_evidence_forgetting_s3 import (
    compatible_negative,
    deterministic_dropout,
    dialogue_fold,
    mask_token_spans,
    replace_dialogue_arguments,
    similarity_bin,
)


class RetrievalEvidenceForgettingS3BuilderTests(unittest.TestCase):
    def test_dialogue_masking_is_case_insensitive_and_role_preserving(self):
        value = replace_dialogue_arguments(
            ["Ross: I met Rachel.", "RACHEL: Hello Ross."], "Ross", "Rachel"
        )
        self.assertNotIn("Ross", value)
        self.assertNotIn("Rachel", value)
        self.assertGreaterEqual(value.count("[ARG1]"), 2)
        self.assertGreaterEqual(value.count("[ARG2]"), 2)

    def test_token_span_masking_supports_reverse_order(self):
        self.assertEqual(
            mask_token_spans("a b c d".split(), (3, 4), (0, 1)),
            "[ARG2] b c [ARG1]",
        )

    def test_dropout_is_deterministic_and_keeps_markers(self):
        text = "one [ARG1] two three [ARG2] four five six seven"
        first = deterministic_dropout(text, "row")
        second = deterministic_dropout(text, "row")
        self.assertEqual(first, second)
        self.assertIn("[ARG1]", first)
        self.assertIn("[ARG2]", first)
        self.assertNotEqual(first, "")

    def test_source_identity_is_content_based(self):
        from tasks.build_retrieval_evidence_forgetting_s3 import source_hash
        self.assertEqual(source_hash("row", "same"), source_hash("row", "same"))
        self.assertNotEqual(source_hash("row", "same"), source_hash("row", "other"))

    def test_negative_compatibility_rejects_multilabel_overlap(self):
        anchor = {"source": "dialogre", "type_pair": "PER:PER",
                  "all_relations_audit": ["dialogre:a", "dialogre:b"]}
        overlap = {"source": "dialogre", "type_pair": "PER:PER",
                   "all_relations_audit": ["dialogre:b", "dialogre:c"]}
        clean = {"source": "dialogre", "type_pair": "PER:PER",
                 "all_relations_audit": ["dialogre:c"]}
        wrong_type = {"source": "dialogre", "type_pair": "PER:GPE",
                      "all_relations_audit": ["dialogre:c"]}
        self.assertFalse(compatible_negative(anchor, overlap))
        self.assertTrue(compatible_negative(anchor, clean))
        self.assertFalse(compatible_negative(anchor, wrong_type))

    def test_fold_and_bin_are_bounded(self):
        self.assertEqual(dialogue_fold("abc"), dialogue_fold("abc"))
        self.assertIn(dialogue_fold("abc"), range(5))
        self.assertEqual(similarity_bin(1.0), 19)


if __name__ == "__main__":
    unittest.main()
