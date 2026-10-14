import json
from pathlib import Path
import tempfile
import unittest

from scripts.audit_controlled_interference_v18 import audit
from tasks.build_controlled_interference_v18 import (
    FIXED_POLICIES, MECHANISMS, build, build_episode, build_states,
)


class ControlledInterferenceV18Test(unittest.TestCase):
    def test_episode_mechanisms_and_causal_states(self):
        for index, mechanism in enumerate(MECHANISMS):
            episode = build_episode(index, 16, mechanism)
            self.assertEqual(len(episode["sessions"]), 16)
            self.assertNotEqual(episode["primary_old_value"],
                                episode["primary_current_value"])
            self.assertEqual(tuple(query["kind"] for query in episode["queries"]),
                             ("update_old_surface", "update_current_surface", "retain_1",
                              "retain_2", "locality"))
            state = build_states(episode)
            self.assertEqual(tuple(state["fixed"]), FIXED_POLICIES)
            self.assertEqual(len(state["delete_one"]), 15)
            self.assertEqual(len(state["delete_key"]), 6)
            self.assertTrue(all(episode["current_session_id"] in row["memory_session_ids"]
                                for row in state["delete_one"] + state["delete_key"]))

    def test_full_build_is_deterministic_balanced_and_audited(self):
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first, second = Path(first_dir) / "v18", Path(second_dir) / "v18"
            manifest = build(first)
            second_manifest = build(second)
            self.assertEqual(manifest["pilot_sha256"], second_manifest["pilot_sha256"])
            self.assertEqual(manifest["pilot_states_sha256"],
                             second_manifest["pilot_states_sha256"])
            self.assertEqual(manifest["counts"]["by_mechanism"],
                             {name: 10 for name in MECHANISMS})
            self.assertEqual(manifest["counts"]["single_deletion_states"], 1160)
            self.assertEqual(manifest["counts"]["key_deletion_states"], 240)
            result = audit(first, first / "audit.json")
            self.assertTrue(result["audit_passed"])
            rows = [json.loads(line) for line in (first / "pilot.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 40)


if __name__ == "__main__":
    unittest.main()
