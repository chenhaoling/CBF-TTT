"""Tests for nested-LOSO ZsRE forgetting controller data flow."""

import unittest

from tasks.train_zsre_forgetting_controller import (
    MAX_KEYS, choose_inner_tune, lexical_features, oracle_event_groups,
)


def session():
    return {"id": "s", "chains": [
        {"id": "main", "kind": "primary", "versions": [
            {"prompt": "Who is Ada's mother?"}, {"prompt": "What is the name of Ada's mother?"}]},
        {"id": "aux", "kind": "auxiliary", "parent_chain_id": "main", "versions": [
            {"prompt": "The occupation of Ada is"}]},
        {"id": "other", "kind": "primary", "versions": [{"prompt": "Where is Bob from?"}]},
    ], "events": [
        {"chain_id": "main", "version": 0, "action": "retain", "kind": "primary_initial"},
        {"chain_id": "aux", "version": 0, "action": "retain", "kind": "hard_retain"},
        {"chain_id": "other", "version": 0, "action": "retain", "kind": "primary_initial"},
        {"chain_id": "main", "version": 1, "action": "local", "kind": "primary_update"},
    ]}


class ControllerTests(unittest.TestCase):
    def test_oracle_groups_encode_retain_and_exact_delete_candidate(self):
        groups = oracle_event_groups([session()])
        self.assertEqual(len(groups), 3)
        self.assertEqual(groups[0]["target"], MAX_KEYS)
        self.assertEqual(groups[1]["target"], MAX_KEYS)
        self.assertEqual(groups[2]["target"], 0)
        self.assertEqual(groups[0]["group_id"], "main")

    def test_feature_code_has_no_chain_metadata_argument(self):
        values = lexical_features("Who is Ada's mother?", "What is the name of Ada's mother?")
        unrelated = lexical_features("The occupation of Ada is", "What is the name of Ada's mother?")
        self.assertEqual(len(values), 6)
        self.assertGreater(values[2], unrelated[2])

    def test_inner_tune_is_deterministic_and_excludes_outer(self):
        sessions = ["a", "b", "c"]
        first = choose_inner_tune(sessions, "held")
        self.assertEqual(first, choose_inner_tune(list(reversed(sessions)), "held"))
        self.assertIn(first, sessions)


if __name__ == "__main__":
    unittest.main()
