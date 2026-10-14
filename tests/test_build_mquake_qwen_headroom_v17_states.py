"""Tests for exhaustive causal single-deletion state freezing."""

import unittest

from tasks.build_mquake_qwen_headroom_v17_states import freeze_states


class HeadroomStateTest(unittest.TestCase):
    def setUp(self):
        self.episode = {"sessions": [
            {"session_id": "d0"}, {"session_id": "old"},
            {"session_id": "hard"}, {"session_id": "current"},
            {"session_id": "future"},
        ]}
        self.sample = {"query_step": 3, "primary_old_session_id": "old",
                       "primary_current_session_id": "current"}

    def test_fixed_and_every_historical_single_deletion(self):
        fixed, deletions = freeze_states(self.episode, self.sample)
        self.assertEqual(fixed["retain_all"], ["d0", "old", "hard", "current"])
        self.assertEqual(fixed["global_reset"], ["current"])
        self.assertEqual(fixed["last_1"], ["current"])
        self.assertEqual(fixed["last_2"], ["hard", "current"])
        self.assertEqual([row["deleted_session_id"] for row in deletions],
                         ["d0", "old", "hard"])
        old = next(row for row in deletions if row["deleted_session_id"] == "old")
        self.assertEqual(old["memory_session_ids"], ["d0", "hard", "current"])

    def test_future_and_current_are_not_deletion_candidates(self):
        _, deletions = freeze_states(self.episode, self.sample)
        deleted = {row["deleted_session_id"] for row in deletions}
        self.assertNotIn("future", deleted)
        self.assertNotIn("current", deleted)
        self.assertTrue(all("current" in row["memory_session_ids"] for row in deletions))

    def test_invalid_query_identity_is_rejected(self):
        with self.assertRaises(ValueError):
            freeze_states(self.episode, dict(self.sample, query_step=2))


if __name__ == "__main__":
    unittest.main()
