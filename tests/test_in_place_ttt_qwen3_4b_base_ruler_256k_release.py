from __future__ import annotations

import json
import os
import platform
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.create_in_place_ttt_qwen3_4b_base_ruler_256k_release_review import (
    MODEL_REVISION,
    build_review,
    sha256,
)
from scripts.create_in_place_ttt_qwen3_4b_base_ruler_256k_release_from_review import (
    CORRECTED_RELEASE_PROTOCOL,
    create_from_review,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "IN_PLACE_TTT_BASELINE_QWEN3_4B_BASE_RULER_256K_RELEASE_ADDENDUM.md"
BUILDER = ROOT / "scripts/create_in_place_ttt_qwen3_4b_base_ruler_256k_release_review.py"
TRANSITION = ROOT / "scripts/create_in_place_ttt_qwen3_4b_base_ruler_256k_release_from_review.py"
RELEASE_TOOL = ROOT / "scripts/create_in_place_ttt_2x5090_ruler_256k_release.py"
WATCHER = ROOT / "scripts/watch_and_prepare_in_place_ttt_qwen3_4b_base_ruler_256k_release_review.sh"


def write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")


def fixtures(root: Path) -> tuple[Path, Path, Path]:
    resource = root / "resource.json"
    write(resource, {
        "protocol": "in_place_ttt_2x5090_ruler_256k_resource_audit_v1",
        "passed": True, "errors": [], "context_length": 262144,
        "extrapolation": True, "scores_discarded": True,
        "formal_256k_released": True,
        "boundaries": {"no_ttt": {}, "ttt": {}},
    })
    corrected = root / "corrected.json"
    write(corrected, {
        "protocol": "in_place_ttt_qwen3_4b_base_ruler_256k_resource_audit_v1",
        "passed": True, "errors": [], "model_revision": MODEL_REVISION,
        "context_length": 262144, "extrapolation": True,
        "scores_discarded": True, "formal_256k_released": True,
        "underlying_resource_audit_sha256": sha256(resource),
    })
    identity = root / "identity.json"
    write(identity, {
        "passed": True, "checks": {"revision": True},
        "revision": MODEL_REVISION,
    })
    return resource, corrected, identity


class CorrectedBaseRuler256KReleaseTest(unittest.TestCase):
    def test_plan_and_watcher_are_non_authorizing(self):
        text = PLAN.read_text()
        self.assertIn("formal_release_created=false", text)
        self.assertIn("explicit `--accept`", text)
        self.assertIn("hku-gpu2", text)
        subprocess.run(["bash", "-n", str(WATCHER)], check=True)
        watcher = WATCHER.read_text()
        self.assertNotIn(" --accept", watcher)
        self.assertIn("completed_ruler_256k_release_review", watcher)
        self.assertIn("CUDA_VISIBLE_DEVICES=", watcher)
        self.assertNotIn('"$PYTHON_BIN" "$TRANSITION"', watcher)

    def test_review_and_explicit_transition_bind_corrected_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            resource, corrected, identity = fixtures(root)
            target, packet = root / "release.json", root / "review.json"
            with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
                review = build_review(
                    resource_audit=resource,
                    corrected_resource_audit=corrected,
                    model_identity=identity, release_tool=RELEASE_TOOL,
                    review_builder=BUILDER, transition_tool=TRANSITION,
                    plan=PLAN, target_release=target, output=packet,
                    expected_hostname=platform.node(),
                )
                self.assertTrue(review["ready_for_human_review"])
                self.assertFalse(review["formal_release_created"])
                self.assertFalse(target.exists())
                with self.assertRaises(PermissionError):
                    create_from_review(
                        review_packet=packet, transition_plan=PLAN,
                        output=target, accept=False, reviewed_by="reviewer",
                        decision_note="resource accepted",
                        expected_hostname=platform.node(),
                    )
                self.assertFalse(target.exists())
                release = create_from_review(
                    review_packet=packet, transition_plan=PLAN,
                    output=target, accept=True, reviewed_by="reviewer",
                    decision_note="resource accepted",
                    expected_hostname=platform.node(),
                )
            self.assertTrue(release["accepted"])
            self.assertEqual(release["corrected_base_release_protocol"],
                             CORRECTED_RELEASE_PROTOCOL)
            self.assertEqual(release["corrected_resource_audit_sha256"],
                             sha256(corrected))
            self.assertEqual(release["review_packet_sha256"], sha256(packet))

    def test_transition_rejects_changed_corrected_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            resource, corrected, identity = fixtures(root)
            target, packet = root / "release.json", root / "review.json"
            with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
                build_review(
                    resource_audit=resource,
                    corrected_resource_audit=corrected,
                    model_identity=identity, release_tool=RELEASE_TOOL,
                    review_builder=BUILDER, transition_tool=TRANSITION,
                    plan=PLAN, target_release=target, output=packet,
                    expected_hostname=platform.node(),
                )
                corrected.write_text(corrected.read_text() + "\n")
                with self.assertRaises(ValueError):
                    create_from_review(
                        review_packet=packet, transition_plan=PLAN,
                        output=target, accept=True, reviewed_by="reviewer",
                        decision_note="resource accepted",
                        expected_hostname=platform.node(),
                    )
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
