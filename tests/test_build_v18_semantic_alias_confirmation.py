import unittest

from tasks.build_v18_semantic_alias_confirmation import (
    ALIAS_BANK, LENGTHS, MECHANISMS, build_episode, build_states,
)


class BuildV18SemanticAliasConfirmationTest(unittest.TestCase):
    def test_balanced_deterministic_source_disjoint_structure(self):
        rows = [build_episode(index) for index in range(320)]
        self.assertEqual({length: sum(row["length"] == length for row in rows)
                          for length in LENGTHS}, {length: 80 for length in LENGTHS})
        self.assertEqual({name: sum(row["mechanism"] == name for row in rows)
                          for name in MECHANISMS}, {name: 80 for name in MECHANISMS})
        self.assertEqual({fold: sum(row["fold"] == fold for row in rows)
                          for fold in range(5)}, {fold: 64 for fold in range(5)})
        self.assertEqual(rows, [build_episode(index) for index in range(320)])
        self.assertEqual(len(ALIAS_BANK), 20)

    def test_primary_surfaces_are_distinct_but_share_latent_slot(self):
        for index in range(320):
            row = build_episode(index); current = row["sessions"][-1]
            self.assertEqual(len({row["source_keys"][0], row["update_key_id"],
                                  row["query_key_id"]}), 3)
            self.assertEqual(current["key_id"], row["update_key_id"])
            self.assertEqual(current["semantic_key_index"], 0)
            self.assertEqual(row["queries"][0]["query_key_id"], row["source_keys"][0])
            self.assertEqual(row["queries"][1]["query_key_id"], row["query_key_id"])

    def test_exact_rule_misses_and_semantic_oracle_removes_conflict(self):
        for index in range(320):
            episode = build_episode(index); state = build_states(episode)
            self.assertEqual(state["exact_removed"], [])
            self.assertTrue(state["semantic_removed"])
            self.assertIn(episode["current_session_id"], state["semantic_oracle"])
            self.assertTrue(all(value not in state["semantic_oracle"]
                                for value in state["semantic_removed"]))

    def test_mechanism_roles_and_current_update(self):
        for index in range(320):
            row = build_episode(index); roles = [value["role"] for value in row["sessions"]]
            self.assertEqual(roles[-1], "current_update")
            expected = {"stale_repetition": "stale_repeat",
                        "irrelevant_near_duplicate": "near_duplicate",
                        "useful_repetition": "useful_repeat"}.get(row["mechanism"])
            if expected:
                self.assertIn(expected, roles)


if __name__ == "__main__":
    unittest.main()
