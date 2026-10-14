"""Tests for the independent V17 causal state reconstruction."""

import unittest

from scripts.audit_mquake_qwen_headroom_v17_states import reconstruct


class HeadroomStateAuditTest(unittest.TestCase):
    def test_independent_reconstruction(self):
        episode = {"sessions": [{"session_id": "a"}, {"session_id": "old"},
                                {"session_id": "current"}, {"session_id": "future"}]}
        sample = {"query_step": 2, "primary_old_session_id": "old",
                  "primary_current_session_id": "current"}
        fixed, deletions = reconstruct(episode, sample)
        self.assertEqual(fixed["last_2"], ["old", "current"])
        self.assertEqual(deletions, [
            {"deleted_session_id": "a", "memory_session_ids": ["old", "current"]},
            {"deleted_session_id": "old", "memory_session_ids": ["a", "current"]},
        ])

    def test_invalid_current_position_rejected(self):
        episode = {"sessions": [{"session_id": "old"}, {"session_id": "x"}]}
        sample = {"query_step": 1, "primary_old_session_id": "old",
                  "primary_current_session_id": "missing"}
        with self.assertRaises(ValueError):
            reconstruct(episode, sample)


if __name__ == "__main__":
    unittest.main()
