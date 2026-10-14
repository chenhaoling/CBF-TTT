import tempfile
from pathlib import Path
import unittest
from unittest import mock

from tasks.build_controlled_interference_v18 import LENGTHS, MECHANISMS
from tasks.build_controlled_interference_v20_alias import (
    ALIAS_BANK, build, build_episode, build_states,
)


class BuildControlledInterferenceV20AliasTest(unittest.TestCase):
    def test_balanced_new_three_surface_design(self):
        rows = [build_episode(index) for index in range(320)]
        self.assertEqual({length: sum(row["length"] == length for row in rows)
                          for length in LENGTHS}, {length: 80 for length in LENGTHS})
        self.assertEqual({name: sum(row["mechanism"] == name for row in rows)
                          for name in MECHANISMS}, {name: 80 for name in MECHANISMS})
        self.assertEqual({fold: sum(row["fold"] == fold for row in rows)
                          for fold in range(5)}, {fold: 64 for fold in range(5)})
        self.assertEqual(len(ALIAS_BANK), 20)
        self.assertEqual(rows, [build_episode(index) for index in range(320)])

    def test_exact_rule_misses_latent_conflict(self):
        for index in range(320):
            episode = build_episode(index); state = build_states(episode)
            self.assertEqual(len({episode["source_keys"][0], episode["update_key_id"],
                                  episode["query_key_id"]}), 3)
            self.assertEqual(state["exact_removed"], [])
            self.assertTrue(state["semantic_removed"])
            self.assertEqual(state["special"]["exact_conflict"],
                             state["fixed"]["retain_all"])
            self.assertIn(episode["current_session_id"],
                          state["special"]["semantic_oracle"])

    def test_build_is_deterministic_and_unscored(self):
        dependency = {"rule_summary_sha256": "a", "rule_audit_sha256": "b",
                      "v19_raw_rows_sha256": {}, "v19_outcome_summary_sha256": "c"}
        with tempfile.TemporaryDirectory() as directory:
            one, two = Path(directory) / "one", Path(directory) / "two"
            with mock.patch(
                    "tasks.build_controlled_interference_v20_alias.verify_release",
                    return_value=dependency):
                first = build(one, "rules"); second = build(two, "rules")
        self.assertEqual(first["alias_sha256"], second["alias_sha256"])
        self.assertEqual(first["states_sha256"], second["states_sha256"])
        self.assertFalse(first["model_scored"])
        self.assertFalse(first["controller_or_threshold_selected"])


if __name__ == "__main__":
    unittest.main()
