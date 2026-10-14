"""Focused tests for the independent V16 pilot audit."""

import hashlib
import unittest

from scripts.audit_mquake_qwen_outcome_v16_pilot import (
    expected_memory, expected_specs, render,
)


class OutcomePilotAuditTest(unittest.TestCase):
    def setUp(self):
        self.episode = {"sessions": [
            {"session_id": "d0", "units": ["d"]},
            {"session_id": "old", "units": ["o"]},
            {"session_id": "hard", "units": ["h"]},
            {"session_id": "current", "units": ["c"]},
            {"session_id": "future", "units": ["f"]},
        ]}
        self.sample = {"query_step": 3, "primary_old_session_id": "old",
                       "primary_current_session_id": "current",
                       "old": {"statement": "old?", "value": "A"},
                       "current": {"statement": "current?", "value": "B"},
                       "hard_retain": {"statement": "hard?", "value": "C"},
                       "unrelated": {"statement": "other?", "value": "D"}}

    def test_independent_memory_replay_is_causal(self):
        self.assertEqual(expected_memory(self.episode, self.sample, "no_memory"), [])
        self.assertEqual([row["session_id"] for row in
                          expected_memory(self.episode, self.sample, "oracle")],
                         ["d0", "hard", "current"])

    def test_independent_specs_and_prompt_hash(self):
        specs = expected_specs(self.sample)
        self.assertEqual([row[0] for row in specs], [
            "primary_old_phrase", "primary_current_phrase", "hard_retain", "unrelated"])
        prompt = render(expected_memory(self.episode, self.sample, "oracle"),
                        {key: f"text-{key}" for key in "dohcf"}, specs[0][1])
        self.assertNotIn("text-o", prompt)
        self.assertNotIn("text-f", prompt)
        self.assertEqual(len(hashlib.sha256(prompt.encode()).hexdigest()), 64)

    def test_future_query_point_is_rejected(self):
        invalid = dict(self.sample, query_step=2)
        with self.assertRaises(ValueError):
            expected_memory(self.episode, invalid, "oracle")


if __name__ == "__main__":
    unittest.main()
