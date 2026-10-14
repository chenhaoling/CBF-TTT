"""Tests for causal formal-state reconstruction from frozen OOF traces."""

import unittest

from tasks.build_mquake_qwen_outcome_v16_formal_states import (
    POLICIES, policy_states, replay_trace,
)


class FormalStateBuilderTest(unittest.TestCase):
    def setUp(self):
        self.sessions = [
            {"session_id": "d0", "event_kind": "distractor"},
            {"session_id": "old", "event_kind": "primary_old"},
            {"session_id": "hard", "event_kind": "hard_retain"},
            {"session_id": "current", "event_kind": "primary_update"},
            {"session_id": "future", "event_kind": "distractor"},
        ]
        self.episode = {"episode_id": "e", "primary_subject_hash": "s", "length": 5,
                        "sessions": self.sessions}
        self.sample = {"query_step": 3, "primary_old_session_id": "old",
                       "primary_current_session_id": "current"}

    def trace(self, deleted=None):
        decisions = []
        for step, event in enumerate(self.sessions):
            decisions.append({"step": step, "event_kind": event["event_kind"],
                              "deleted": deleted if step == 3 else None})
        return {"episode_id": "e", "subject": "s", "length": 5,
                "decisions": decisions}

    def test_replay_deletes_before_admitting_current(self):
        memory = replay_trace(self.episode, self.trace("old"), 3)
        self.assertEqual([row["session_id"] for row in memory], ["d0", "hard", "current"])

    def test_policy_matrix_is_causal_and_ordered(self):
        states = policy_states(self.episode, self.sample, {
            "v14": self.trace(), "v15": self.trace("old"),
            "matched_random": self.trace("d0")})
        self.assertEqual(tuple(states), POLICIES)
        self.assertEqual(states["retain_all"], ["d0", "old", "hard", "current"])
        self.assertEqual(states["global_reset"], ["current"])
        self.assertEqual(states["last_1"], ["current"])
        self.assertEqual(states["last_2"], ["hard", "current"])
        self.assertEqual(states["oracle"], ["d0", "hard", "current"])
        self.assertTrue(all("future" not in values for values in states.values()))

    def test_nonvisible_deletion_is_rejected(self):
        with self.assertRaises(ValueError):
            replay_trace(self.episode, self.trace("future"), 3)

    def test_trace_identity_is_checked(self):
        trace = self.trace(); trace["subject"] = "wrong"
        with self.assertRaises(ValueError):
            replay_trace(self.episode, trace, 3)


if __name__ == "__main__":
    unittest.main()
