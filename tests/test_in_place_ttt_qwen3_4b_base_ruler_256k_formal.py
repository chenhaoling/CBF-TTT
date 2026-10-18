from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.audit_in_place_ttt_qwen3_4b_base_ruler_256k import (
    MODEL_REVISION,
    audit,
    sha256,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "IN_PLACE_TTT_BASELINE_QWEN3_4B_BASE_RULER_256K_FORMAL_ADDENDUM.md"
WATCHER = ROOT / "scripts/watch_and_run_in_place_ttt_qwen3_4b_base_ruler_256k.sh"


def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")


def touch(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value + "\n")
    return path


def complete_fixture(base: Path) -> argparse.Namespace:
    root = base / "formal"
    training, matrix, resource = (
        base / "training", base / "matrix", base / "resource"
    )
    tokenizer = base / "tokenizer"
    for path in (root, training, matrix, resource, tokenizer):
        path.mkdir(parents=True)
    training_audit = training / "independent_audit.json"
    write(training_audit, {
        "protocol": "in_place_ttt_2x5090_100m_stage2_audit_v1",
        "passed": True, "errors": [],
    })
    corrected_training = base / "corrected_training.json"
    write(corrected_training, {
        "protocol": "in_place_ttt_qwen3_4b_base_100m_stage2_audit_v1",
        "passed": True, "errors": [], "model_revision": MODEL_REVISION,
        "underlying_formal_audit_sha256": sha256(training_audit),
    })
    matrix_audit = matrix / "independent_matrix_audit.json"
    write(matrix_audit, {
        "protocol": "in_place_ttt_2x5090_ruler_matrix_audit_v1",
        "passed": True, "errors": [],
    })
    corrected_matrix = base / "corrected_matrix.json"
    write(corrected_matrix, {
        "protocol": "in_place_ttt_qwen3_4b_base_ruler_matrix_audit_v1",
        "passed": True, "errors": [], "model_revision": MODEL_REVISION,
        "underlying_matrix_audit_sha256": sha256(matrix_audit),
        "corrected_training_audit_sha256": sha256(corrected_training),
    })
    resource_audit = resource / "independent_resource_audit.json"
    write(resource_audit, {
        "protocol": "in_place_ttt_2x5090_ruler_256k_resource_audit_v1",
        "passed": True, "errors": [], "formal_256k_released": True,
    })
    corrected_resource = base / "corrected_resource.json"
    write(corrected_resource, {
        "protocol": "in_place_ttt_qwen3_4b_base_ruler_256k_resource_audit_v1",
        "passed": True, "errors": [], "model_revision": MODEL_REVISION,
        "underlying_resource_audit_sha256": sha256(resource_audit),
        "corrected_training_audit_sha256": sha256(corrected_training),
        "corrected_matrix_audit_sha256": sha256(corrected_matrix),
    })

    assets = {}
    for name in ("tokenizer.json", "tokenizer_config.json"):
        path = touch(tokenizer / name, name)
        assets[name] = {"size": path.stat().st_size, "sha256": sha256(path)}
    identity = base / "identity.json"
    write(identity, {
        "passed": True, "checks": {"revision": True},
        "revision": MODEL_REVISION, "files": assets,
    })
    names = (
        "public_corpus_plan evaluation_plan extrapolation_plan formal_plan "
        "corrected_256k_plan release_addendum formal_addendum "
        "paper_inference_addendum runner underlying_auditor single_runner "
        "summarizer review_builder transition_tool release_tool"
    ).split()
    files = {name: touch(base / name, name) for name in names}
    paper_hash = sha256(files["paper_inference_addendum"])
    for arm in ("no_ttt", "ttt"):
        write(root / arm / "independent_audit.json", {
            "protocol": "in_place_ttt_2x5090_ruler_audit_v1",
            "passed": True, "errors": [], "arm": arm,
            "context_length": 262144,
            "bindings": {"paper_inference_addendum_sha256": paper_hash},
        })
    scores = {"no_ttt": 40.0, "ttt": 43.0, "ttt_minus_no_ttt": 3.0}
    write(root / "independent_audit.json", {
        "protocol": "in_place_ttt_2x5090_ruler_256k_audit_v1",
        "passed": True, "errors": [], "context_length": 262144,
        "claim_boundary": "public_data_yarn_extrapolation_not_paper_exact",
        "scores": scores,
    })
    write(root / "summary.json", {
        "protocol": "in_place_ttt_2x5090_ruler_256k_summary_v1",
        "completed": True, "errors": [], "scores": scores,
    })
    release = base / "release.json"
    review_packet = base / "review.json"
    review_bindings = {
        "resource_audit_sha256": sha256(resource_audit),
        "corrected_resource_audit_sha256": sha256(corrected_resource),
        "model_identity_sha256": sha256(identity),
        "release_tool_sha256": sha256(files["release_tool"]),
        "review_builder_sha256": sha256(files["review_builder"]),
        "transition_tool_sha256": sha256(files["transition_tool"]),
        "plan_sha256": sha256(files["release_addendum"]),
    }
    write(review_packet, {
        "protocol": "in_place_ttt_qwen3_4b_base_ruler_256k_release_review_v1",
        "ready_for_human_review": True, "formal_release_created": False,
        "target_release_path": str(release.resolve()),
        "required_human_fields": ["accept", "reviewed_by", "decision_note"],
        "bindings": review_bindings,
        "review": {
            "resource_audit_sha256": sha256(resource_audit),
            "corrected_resource_audit_sha256": sha256(corrected_resource),
            "model_identity_sha256": sha256(identity),
            "model_revision": MODEL_REVISION,
        },
    })
    write(release, {
        "protocol": "in_place_ttt_2x5090_ruler_256k_formal_release_v1",
        "corrected_base_release_protocol":
            "in_place_ttt_qwen3_4b_base_ruler_256k_formal_release_v1",
        "accepted": True, "model_revision": MODEL_REVISION,
        "resource_audit_sha256": sha256(resource_audit),
        "corrected_resource_audit_sha256": sha256(corrected_resource),
        "review_packet_sha256": sha256(review_packet),
        "release_transition_plan_sha256": sha256(files["release_addendum"]),
        "reviewed_by": "reviewer", "decision_note": "accepted",
    })
    write(root / "formal_metadata.json", {
        "protocol": "in_place_ttt_2x5090_ruler_256k_formal_run_v1",
        "hostname": "user-Super-Server",
        "training_root": str(training.resolve()),
        "matrix_root": str(matrix.resolve()),
        "resource_root": str(resource.resolve()),
        "training_audit_sha256": sha256(training_audit),
        "matrix_audit_sha256": sha256(matrix_audit),
        "resource_audit_sha256": sha256(resource_audit),
        "release_sha256": sha256(release),
        **{
            f"{name}_sha256": sha256(files[name])
            for name in (
                "public_corpus_plan", "evaluation_plan", "extrapolation_plan",
                "formal_plan", "runner", "single_runner", "summarizer",
                "underlying_auditor",
            )
        },
        "auditor_sha256": sha256(files["underlying_auditor"]),
    })
    # The generic metadata key is auditor_sha256, not underlying_auditor_sha256.
    metadata = json.loads((root / "formal_metadata.json").read_text())
    metadata.pop("underlying_auditor_sha256")
    write(root / "formal_metadata.json", metadata)
    return argparse.Namespace(
        root=root, training_root=training,
        corrected_training_audit=corrected_training, matrix_root=matrix,
        corrected_matrix_audit=corrected_matrix, resource_root=resource,
        corrected_resource_audit=corrected_resource,
        review_packet=review_packet, release=release, model_identity=identity,
        tokenizer=tokenizer, output=base / "output.json", **files,
    )


class CorrectedBaseRuler256KFormalTest(unittest.TestCase):
    def test_plan_and_watcher_preserve_release_and_paper_overlay(self):
        text = PLAN.read_text()
        self.assertIn("hku-gpu2", text)
        self.assertIn("100 examples per task", text)
        self.assertIn("tau=1e-5", text)
        self.assertIn("does not release update-admission", text)
        subprocess.run(["bash", "-n", str(WATCHER)], check=True)
        watcher = WATCHER.read_text()
        self.assertIn("waiting_for_accepted_corrected_ruler_256k_release", watcher)
        self.assertIn('PAPER_INFERENCE_ADDENDUM="$PAPER_INFERENCE_ADDENDUM"', watcher)
        self.assertIn("completed_corrected_formal_ruler_256k", watcher)
        self.assertNotIn(" --accept", watcher)

    def test_auditor_fails_closed_when_inputs_are_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fields = (
                "root training_root corrected_training_audit matrix_root "
                "corrected_matrix_audit resource_root corrected_resource_audit "
                "review_packet release model_identity tokenizer public_corpus_plan "
                "evaluation_plan extrapolation_plan formal_plan corrected_256k_plan "
                "release_addendum formal_addendum paper_inference_addendum runner "
                "underlying_auditor single_runner summarizer review_builder "
                "transition_tool release_tool output"
            ).split()
            args = argparse.Namespace(**{name: root / name for name in fields})
            result = audit(args)
            self.assertFalse(result["passed"])
            self.assertIn("missing required files", result["errors"][0])

    def test_update_admission_remains_false_in_source(self):
        source = (ROOT / "scripts/audit_in_place_ttt_qwen3_4b_base_ruler_256k.py").read_text()
        self.assertIn('"update_admission_released": False', source)

    def test_complete_wrapper_passes_and_paper_overlay_drift_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = complete_fixture(Path(temporary))
            passed = audit(args)
            self.assertTrue(passed["passed"], passed["errors"])
            args.paper_inference_addendum.write_text("changed\n")
            failed = audit(args)
            self.assertFalse(failed["passed"])
            self.assertTrue(any("paper overlay" in error
                                for error in failed["errors"]))


if __name__ == "__main__":
    unittest.main()
