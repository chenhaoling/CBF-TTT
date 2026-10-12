"""Pure construction tests for the cross-domain forgetting reranker."""

import unittest

from tasks.train_crossdomain_forgetting_reranker import deterministic_folds, tag_sessions


class CrossDomainForgettingRerankerTests(unittest.TestCase):
    def test_folds_are_deterministic_and_balanced(self):
        ids = [f"session-{index}" for index in range(23)]
        first = deterministic_folds(ids); second = deterministic_folds(reversed(ids))
        self.assertEqual(first, second)
        counts = [sum(value == fold for value in first.values()) for fold in range(5)]
        self.assertLessEqual(max(counts) - min(counts), 1)

    def test_tag_sessions_updates_references_without_visible_text(self):
        sessions = [{"id": "s", "chains": [
            {"id": "p", "versions": [{"prompt": "visible"}]},
            {"id": "a", "parent_chain_id": "p", "versions": []}],
            "events": [{"chain_id": "p"}, {"chain_id": "a"}]}]
        tagged = tag_sessions(sessions, "source")
        self.assertEqual(tagged[0]["id"], "source:s")
        self.assertEqual(tagged[0]["chains"][1]["parent_chain_id"], "source:p")
        self.assertEqual(tagged[0]["events"][0]["chain_id"], "source:p")
        self.assertEqual(tagged[0]["chains"][0]["versions"][0]["prompt"], "visible")
        self.assertEqual(sessions[0]["id"], "s")


if __name__ == "__main__":
    unittest.main()
