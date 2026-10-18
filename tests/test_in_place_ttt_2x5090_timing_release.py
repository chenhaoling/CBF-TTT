from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.create_in_place_ttt_2x5090_timing_release import (
    create_release,
    review,
    sha256,
)


def timing_audit(path: Path, stage: str) -> Path:
    suffix = "stage1" if stage == "stage1" else "stage2"
    arm_key = f"projected_formal_{suffix}_seconds"
    path.write_text(json.dumps({
        "protocol": f"in_place_ttt_2x5090_{suffix}_timing_audit_v1",
        "passed": True,
        "errors": [],
        "arms": {
            "no_ttt": {
                arm_key: 270.0,
                "warm_excluded_median_seconds": 10.0,
            },
            "ttt": {
                arm_key: 324.0,
                "warm_excluded_median_seconds": 12.0,
            },
        },
        f"projected_sequential_{suffix}_seconds": 594.0,
    }))
    return path


class TwoRTX5090TimingReleaseTest(unittest.TestCase):
    def test_stage1_explicit_accept_writes_bound_release(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audit = timing_audit(root / "audit.json", "stage1")
            output = root / "release.json"
            result = create_release(
                stage="stage1", timing_audit=audit, output=output,
                accept=True, reviewed_by="tester", decision_note="finite pilot",
            )
            saved = json.loads(output.read_text())
            self.assertEqual(
                result["protocol"], "in_place_ttt_2x5090_100m_stage1_release_v1"
            )
            self.assertTrue(saved["accepted"])
            self.assertEqual(saved["timing_audit_sha256"], sha256(audit))
            self.assertAlmostEqual(saved["projected_sequential_hours"], 0.165)

    def test_missing_explicit_accept_writes_nothing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audit = timing_audit(root / "audit.json", "stage1")
            output = root / "release.json"
            with self.assertRaises(PermissionError):
                create_release(
                    stage="stage1", timing_audit=audit, output=output,
                    accept=False, reviewed_by="tester", decision_note="review",
                )
            self.assertFalse(output.exists())
            self.assertFalse(output.with_name("release.json.partial").exists())

    def test_invalid_or_inconsistent_projection_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            audit = timing_audit(Path(temporary) / "audit.json", "stage1")
            data = json.loads(audit.read_text())
            data["projected_sequential_stage1_seconds"] = 1.0
            audit.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "does not equal"):
                review("stage1", audit)

    def test_stage2_uses_distinct_protocol(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audit = timing_audit(root / "audit.json", "stage2")
            output = root / "release.json"
            result = create_release(
                stage="stage2", timing_audit=audit, output=output,
                accept=True, reviewed_by="tester", decision_note="finite pilot",
            )
            self.assertEqual(
                result["protocol"], "in_place_ttt_2x5090_100m_stage2_release_v1"
            )


if __name__ == "__main__":
    unittest.main()
