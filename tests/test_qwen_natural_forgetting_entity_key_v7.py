import unittest

from tasks.train_qwen_natural_forgetting_entity_key_v7 import (
    ANCHOR_STOPWORDS,
    ENTITY_FEATURES,
    FEATURE_NAMES,
    entity_anchors,
    entity_features,
)


class EntityKeyV7Test(unittest.TestCase):
    def test_same_relation_different_subject_fails_barrier(self):
        incoming = "The located in the administrative territorial entity of Ijdan is"
        top = "The located in the administrative territorial entity of Rzechówek is"
        values, barrier, left, right = entity_features(incoming, top)
        self.assertFalse(barrier)
        self.assertEqual(left, ("ijdan",))
        self.assertEqual(right, ("rzechówek",))
        self.assertEqual(values[2], 0.0)

    def test_paraphrase_same_subject_passes_barrier(self):
        incoming = "Which State is located in Rzechówek?"
        top = "What state is Rzechówek located?"
        values, barrier, left, right = entity_features(incoming, top)
        self.assertTrue(barrier)
        self.assertEqual(left, ("rzechówek", "state"))
        self.assertEqual(right, ("rzechówek",))
        self.assertEqual(values[2], 1.0)

    def test_empty_anchor_never_passes(self):
        values, barrier, _, _ = entity_features("what is this", "which is this")
        self.assertFalse(barrier)
        self.assertEqual(values[0], 0.0)

    def test_feature_registration_and_stopwords(self):
        self.assertEqual(len(FEATURE_NAMES), 15)
        self.assertEqual(tuple(FEATURE_NAMES[-5:]), ENTITY_FEATURES)
        self.assertIn("the", ANCHOR_STOPWORDS)
        self.assertEqual(entity_anchors("The Ijdan is"), {"ijdan"})
        self.assertFalse({"subject", "chain_id", "source_action"} & set(FEATURE_NAMES))


if __name__ == "__main__":
    unittest.main()
