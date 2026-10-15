import unittest

from tasks.evaluate_qwen_natural_scope_v5_development import build_trace


class NaturalScopeV5DevelopmentTest(unittest.TestCase):
    def test_oracle_trace_has_registered_checkpoints(self):
        primary = {"id": "p", "kind": "primary", "subject": "s",
                   "versions": [
                       {"version": 0, "prompt": "p0", "rephrase_prompt": "p0r", "target": "a0"},
                       {"version": 1, "prompt": "p1", "rephrase_prompt": "p1r", "target": "a1"},
                   ]}
        auxiliary = {"id": "a", "kind": "auxiliary", "subject": "s",
                     "parent_chain_id": "p", "versions": [
                         {"version": 0, "prompt": "q0", "rephrase_prompt": "q0r", "target": "b"}
                     ]}
        session = {"id": "session", "split": "dev", "chains": [primary, auxiliary],
                   "events": [
                       {"chain_id": "p", "version": 0, "action": "retain", "kind": "primary_initial"},
                       {"chain_id": "a", "version": 0, "action": "retain", "kind": "hard_retain"},
                       {"chain_id": "p", "version": 1, "action": "local", "kind": "primary_update"},
                   ], "candidates": ["a0", "a1", "b"]}
        rows, state = build_trace(session)
        self.assertEqual([row["phase"] for row in state["snapshots"]], ["initial", "update_1"])
        self.assertEqual(len(rows), 8)
        self.assertFalse(any(row["final"] for row in rows[:4]))
        self.assertTrue(all(row["final"] for row in rows[4:]))
        self.assertEqual([row["target"] for row in state["snapshots"][-1]["states"]["oracle"]],
                         ["b", "a1"])


if __name__ == "__main__":
    unittest.main()
