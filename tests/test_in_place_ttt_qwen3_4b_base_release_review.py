from __future__ import annotations

import json
import os
import platform
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.create_in_place_ttt_qwen3_4b_base_release_review import build_review, sha256
from scripts.create_in_place_ttt_qwen3_4b_base_release_from_review import create_from_review


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "IN_PLACE_TTT_BASELINE_QWEN3_4B_BASE_RELEASE_REVIEW_ADDENDUM.md"
WATCHER = ROOT / "scripts/watch_and_prepare_in_place_ttt_qwen3_4b_base_release_review.sh"
TRANSITION_PLAN = ROOT / "IN_PLACE_TTT_BASELINE_QWEN3_4B_BASE_RELEASE_TRANSITION_ADDENDUM.md"
TOOLS = {
    "stage1": ROOT / "scripts/create_in_place_ttt_qwen3_4b_base_stage1_release.py",
    "stage2": ROOT / "scripts/create_in_place_ttt_qwen3_4b_base_stage2_release.py",
}
MODEL_REVISION = "906bfd4b4dc7f14ee4320094d8b41684abff8539"


def fixtures(root: Path, stage: str) -> tuple[Path, Path, Path]:
    number = "1" if stage == "stage1" else "2"
    projection = 300.0 if stage == "stage1" else 500.0
    timing = root / f"{stage}_timing.json"
    timing.write_text(json.dumps({
        "protocol": f"in_place_ttt_2x5090_stage{number}_timing_audit_v1",
        "passed": True, "errors": [],
        f"projected_sequential_stage{number}_seconds": projection,
        "arms": {
            arm: {f"projected_formal_stage{number}_seconds": projection / 2,
                  "warm_excluded_median_seconds": 5.0}
            for arm in ("no_ttt", "ttt")
        },
    }))
    corrected = root / f"{stage}_corrected.json"
    corrected.write_text(json.dumps({
        "protocol": f"in_place_ttt_qwen3_4b_base_stage{number}_timing_audit_v1",
        "passed": True, "errors": [], "model_revision": MODEL_REVISION,
        "formal_release_created": False,
        "underlying_timing_audit_sha256": sha256(timing),
        f"projected_sequential_stage{number}_seconds": projection,
    }))
    stream_name = "stage1_32k" if stage == "stage1" else "stage2_128k"
    records, tokens = ((1728, 56_623_104) if stage == "stage1"
                       else (320, 41_943_040))
    selected = root / f"{stage}_selected.json"
    selected.write_text(json.dumps({
        "protocol": "in_place_ttt_qwen3_4b_base_100m_selected_stream_audit_v1",
        "passed": True, "errors": [], "checks": {"all": True},
        "bindings": {"corpus_manifest_sha256": "a" * 64},
        "independent_stages": {
            stream_name: {
                "selected_records": records,
                "selected_training_tokens": tokens,
                "ordered_stage_indices_sha256": "b" * 64,
                "ordered_training_hashes_sha256": "c" * 64,
            }
        },
    }))
    return timing, corrected, selected


class CorrectedReleaseReviewTest(unittest.TestCase):
    def test_plan_and_watcher_preserve_human_release(self):
        text = PLAN.read_text()
        transition_text = TRANSITION_PLAN.read_text()
        self.assertIn("hku-gpu2", text)
        self.assertIn("formal_release_created=false", text)
        self.assertIn("nonempty `--reviewed-by`", text)
        self.assertIn("never called by a watcher", transition_text)
        subprocess.run(["bash", "-n", str(WATCHER)], check=True)
        watcher = WATCHER.read_text()
        self.assertNotIn(" --accept", watcher)
        self.assertNotIn(
            "create_in_place_ttt_qwen3_4b_base_release_from_review.py", watcher
        )
        self.assertIn("completed_${STAGE}_release_review", watcher)
        self.assertIn("CUDA_VISIBLE_DEVICES=", watcher)

    def test_both_stage_packets_bind_inputs_without_release(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
                for stage in ("stage1", "stage2"):
                    timing, corrected, selected = fixtures(root, stage)
                    target = root / f"{stage}_release.json"
                    output = root / f"{stage}_review.json"
                    packet = build_review(
                        stage=stage, timing_audit=timing,
                        corrected_timing_audit=corrected,
                        selected_stream_audit=selected,
                        release_tool=TOOLS[stage], plan=PLAN,
                        target_release=target, output=output,
                        expected_hostname=platform.node(),
                    )
                    self.assertTrue(packet["ready_for_human_review"])
                    self.assertFalse(packet["formal_release_created"])
                    self.assertFalse(target.exists())
                    self.assertEqual(packet["bindings"]["timing_audit_sha256"],
                                     sha256(timing))
                    self.assertEqual(packet["review"]["selected_stream_stage"],
                                     "stage1_32k" if stage == "stage1"
                                     else "stage2_128k")

    def test_existing_release_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            timing, corrected, selected = fixtures(root, "stage1")
            target = root / "release.json"
            target.write_text("{}")
            with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
                with self.assertRaises(FileExistsError):
                    build_review(
                        stage="stage1", timing_audit=timing,
                        corrected_timing_audit=corrected,
                        selected_stream_audit=selected,
                        release_tool=TOOLS["stage1"], plan=PLAN,
                        target_release=target, output=root / "review.json",
                        expected_hostname=platform.node(),
                    )

    def test_reviewed_transition_still_requires_explicit_accept(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            timing, corrected, selected = fixtures(root, "stage1")
            target = root / "release.json"
            packet_path = root / "review.json"
            with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
                build_review(
                    stage="stage1", timing_audit=timing,
                    corrected_timing_audit=corrected,
                    selected_stream_audit=selected,
                    release_tool=TOOLS["stage1"], plan=PLAN,
                    target_release=target, output=packet_path,
                    expected_hostname=platform.node(),
                )
                with self.assertRaises(PermissionError):
                    create_from_review(
                        review_packet=packet_path, transition_plan=TRANSITION_PLAN,
                        output=target, accept=False, reviewed_by="reviewer",
                        decision_note="projection accepted",
                        expected_hostname=platform.node(),
                    )
                self.assertFalse(target.exists())
                release = create_from_review(
                    review_packet=packet_path, transition_plan=TRANSITION_PLAN,
                    output=target, accept=True, reviewed_by="reviewer",
                    decision_note="projection accepted",
                    expected_hostname=platform.node(),
                )
            self.assertTrue(release["accepted"])
            self.assertEqual(release["review_packet_sha256"], sha256(packet_path))
            self.assertEqual(release["release_transition_plan_sha256"],
                             sha256(TRANSITION_PLAN))

    def test_reviewed_transition_rejects_changed_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            timing, corrected, selected = fixtures(root, "stage2")
            target = root / "release.json"
            packet_path = root / "review.json"
            with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
                build_review(
                    stage="stage2", timing_audit=timing,
                    corrected_timing_audit=corrected,
                    selected_stream_audit=selected,
                    release_tool=TOOLS["stage2"], plan=PLAN,
                    target_release=target, output=packet_path,
                    expected_hostname=platform.node(),
                )
                selected.write_text(selected.read_text() + "\n")
                with self.assertRaises(ValueError):
                    create_from_review(
                        review_packet=packet_path, transition_plan=TRANSITION_PLAN,
                        output=target, accept=True, reviewed_by="reviewer",
                        decision_note="projection accepted",
                        expected_hostname=platform.node(),
                    )
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
