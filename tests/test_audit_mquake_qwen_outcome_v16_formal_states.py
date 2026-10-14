"""Tests for the independent V16 formal-state replay helpers."""

import unittest

from scripts.audit_mquake_qwen_outcome_v16_formal_states import (
    expected_states, trace_state,
)


class FormalStateAuditTest(unittest.TestCase):
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

    def trace(self, deletion=None):
        return {"episode_id": "e", "subject": "s", "length": 5,
                "decisions": [{"step": step, "event_kind": row["event_kind"],
                               "deleted": deletion if step == 3 else None}
                              for step, row in enumerate(self.sessions)]}

    def test_independent_trace_state(self):
        self.assertEqual(trace_state(self.episode, self.trace("old"), 3),
                         ["d0", "hard", "current"])

    def test_independent_policy_states(self):
        values = expected_states(self.episode, self.sample, self.trace(),
                                 self.trace("old"), self.trace("d0"))
        self.assertEqual(values["v14"], ["d0", "old", "hard", "current"])
        self.assertEqual(values["v15"], ["d0", "hard", "current"])
        self.assertEqual(values["matched_random"], ["old", "hard", "current"])
        self.assertEqual(values["oracle"], ["d0", "hard", "current"])

    def test_invalid_trace_chronology_is_rejected(self):
        trace = self.trace(); trace["decisions"][2]["step"] = 9
        with self.assertRaises(ValueError):
            trace_state(self.episode, trace, 3)


if __name__ == "__main__":
    unittest.main()
