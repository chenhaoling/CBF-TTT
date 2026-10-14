import tempfile
from pathlib import Path
import unittest
from unittest import mock

from tasks.build_controlled_interference_v18 import MECHANISMS, opaque_value
from tasks.build_controlled_interference_v19 import build, build_episode


class ControlledInterferenceV19Test(unittest.TestCase):
    def test_registered_balance_and_folds(self):
        rows = [build_episode(index) for index in range(320)]
        self.assertEqual(len({row["subject"] for row in rows}), 80)
        self.assertEqual(len({row["template_group"] for row in rows}), 20)
        self.assertEqual({row["fold"] for row in rows}, set(range(5)))
        for subject_index in range(80):
            group = rows[subject_index * 4:(subject_index + 1) * 4]
            self.assertEqual([row["mechanism"] for row in group], list(MECHANISMS))
            self.assertEqual(len({row["subject"] for row in group}), 1)
            self.assertEqual(len({row["template_group"] for row in group}), 1)
            self.assertEqual({row["length"] for row in group}, {8, 16, 32, 64})

    def test_values_and_surfaces_are_new(self):
        rows = [build_episode(index) for index in range(320)]
        values = [value for row in rows for value in
                  row["initial_values"] + [row["primary_current_value"],
                                            row["locality_value"]]]
        self.assertEqual(len(values), len(set(values)))
        self.assertNotIn(opaque_value(4096), values)
        self.assertTrue(all(row["entity"].startswith("ENT-C-Y") for row in rows))
        self.assertTrue(all(row["template_group"].startswith("v19_templates_")
                            for row in rows))

    def test_build_is_deterministic_and_frozen(self):
        dependencies = {
            "release_summary_sha256": "a", "release_audit_sha256": "b",
            "v18_manifest_sha256": "c", "v18_formal_sha256": "d",
            "v18_pilot_manifest_sha256": "e", "v18_pilot_sha256": "f",
        }
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "first", Path(directory) / "second"
            with mock.patch(
                    "tasks.build_controlled_interference_v19.verify_release",
                    return_value=dependencies):
                one = build(first, "release", "formal", "pilot")
                two = build(second, "release", "formal", "pilot")
        self.assertEqual(one["confirmation_sha256"], two["confirmation_sha256"])
        self.assertEqual(one["states_sha256"], two["states_sha256"])
        self.assertEqual(one["primary_locality_budget"], .03)
        self.assertEqual(one["maximum_locality_failures"], 9)
        self.assertFalse(one["model_scored"])


if __name__ == "__main__":
    unittest.main()
