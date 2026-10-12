"""Construction tests for structured scope supervision."""

import unittest

from tasks.build_scoped_delta_corpus import MAX_KEYS, decision_groups, mask_subject


class ScopedDeltaCorpusTests(unittest.TestCase):
    def test_mask_subject_is_case_insensitive(self):
        value, found = mask_subject("Where was Ada Lovelace born?", "ada lovelace")
        self.assertTrue(found)
        self.assertEqual(value, "Where was <ENTITY> born?")

    def test_decision_groups_label_hard_negative(self):
        sessions = [{"id": "s", "chains": [
            {"id": "p", "subject": "Ada", "versions": [
                {"prompt": "Ada was born in", "source_row_sha256": "p0"},
                {"prompt": "Birthplace of Ada", "source_row_sha256": "p1"}]},
            {"id": "a", "subject": "Ada", "parent_chain_id": "p", "versions": [
                {"prompt": "Ada worked as", "source_row_sha256": "a0"}]},
        ], "events": [
            {"chain_id": "p", "version": 0, "action": "retain", "kind": "primary_initial"},
            {"chain_id": "a", "version": 0, "action": "retain", "kind": "hard_retain"},
            {"chain_id": "p", "version": 1, "action": "local", "kind": "primary_update"},
        ]}]
        groups = decision_groups(sessions, "toy", "dev")
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0]["target"], MAX_KEYS)
        update = groups[1]
        self.assertEqual(update["target"], 0)
        self.assertEqual([row["negative_type"] for row in update["candidates"]],
                         ["same_chain_positive", "same_subject_different_chain"])
        self.assertEqual([row["label"] for row in update["candidates"]], [1, 0])
        self.assertEqual(update["query_source_row_sha256"], "p1")
        self.assertEqual(update["candidates"][0]["source_row_sha256"], "p0")

    def test_chain_identity_is_not_encoded_in_visible_fields(self):
        text, _ = mask_subject("Grace Hopper created", "Grace Hopper")
        self.assertNotIn("chain", text.lower())


if __name__ == "__main__":
    unittest.main()
