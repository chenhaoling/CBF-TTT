import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.audit_controlled_interference_v18_formal import audit
from tasks.build_controlled_interference_v18_formal import (
    MECHANISMS, build, build_formal_episode,
)


class ControlledInterferenceV18FormalTest(unittest.TestCase):
    def test_subject_mechanisms_lengths_templates_and_folds(self):
        rows = [build_formal_episode(index) for index in range(320)]
        self.assertEqual(len({row["subject"] for row in rows}), 80)
        self.assertEqual(len({row["template_group"] for row in rows}), 20)
        self.assertEqual({row["fold"] for row in rows}, set(range(5)))
        for subject_index in range(80):
            group = rows[subject_index * 4:(subject_index + 1) * 4]
            self.assertEqual([row["mechanism"] for row in group], list(MECHANISMS))
            self.assertEqual(len({row["subject"] for row in group}), 1)
            self.assertEqual(len({row["template_group"] for row in group}), 1)
            self.assertEqual(len({row["fold"] for row in group}), 1)
            self.assertEqual({row["length"] for row in group}, {8, 16, 32, 64})

    def test_build_is_deterministic_balanced_and_audited(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            pilot_root, pilot_data = base / "pilot_result", base / "pilot_data"
            pilot_root.mkdir(); pilot_data.mkdir()
            # The gate itself is unit tested through a frozen dependency stub;
            # the independent audit still reconstructs every formal invariant.
            dependencies = {
                "pilot_summary_sha256": "a", "pilot_audit_sha256": "b",
                "pilot_raw_rows_sha256": {}, "pilot_data_manifest_sha256": "c",
                "pilot_data_audit_sha256": "d", "pilot_data_sha256": "e",
                "pilot_states_sha256": "f",
            }
            first, second = base / "first", base / "second"
            with mock.patch(
                    "tasks.build_controlled_interference_v18_formal.verify_pilot_release",
                    return_value=dependencies):
                first_manifest = build(first, pilot_root, pilot_data)
                second_manifest = build(second, pilot_root, pilot_data)
            self.assertEqual(first_manifest["formal_sha256"], second_manifest["formal_sha256"])
            self.assertEqual(first_manifest["formal_states_sha256"],
                             second_manifest["formal_states_sha256"])
            self.assertEqual(first_manifest["counts"]["rows"], 320)
            self.assertEqual(first_manifest["counts"]["by_mechanism"],
                             {name: 80 for name in MECHANISMS})

    def test_value_namespace_is_disjoint_from_pilot(self):
        rows = [build_formal_episode(index) for index in range(320)]
        values = [value for row in rows for value in
                  row["initial_values"] + [row["primary_current_value"],
                                            row["locality_value"]]]
        self.assertEqual(len(values), len(set(values)))
        self.assertNotIn("Z A A A A", values)


if __name__ == "__main__":
    unittest.main()
