from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_in_place_ttt_2x5090_ruler import evaluation_input_sha256
from scripts.audit_in_place_ttt_2x5090_ruler_256k import audit as audit_formal
from scripts.audit_in_place_ttt_2x5090_ruler_256k_resource import audit
from scripts.create_in_place_ttt_2x5090_ruler_256k_release import create_release
from scripts.summarize_in_place_ttt_2x5090_ruler_256k import summarize
from scripts.summarize_in_place_ttt_level_b_ruler import TASK_PREFIXES
from tests.test_in_place_ttt_level_b_ruler import sha, write_json


class TwoRTX5090Ruler256KResourceTest(unittest.TestCase):
    def test_paired_boundary_passes_and_input_drift_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "resource"
            training = Path(temporary) / "training"
            matrix = Path(temporary) / "matrix"
            training.mkdir()
            matrix.mkdir()
            training_audit = training / "independent_audit.json"
            matrix_audit = matrix / "independent_matrix_audit.json"
            write_json(training_audit, {
                "protocol": "in_place_ttt_2x5090_100m_stage2_audit_v1",
                "passed": True,
                "errors": [],
                "arms": {"no_ttt": {}, "ttt": {}},
            })
            write_json(matrix_audit, {
                "protocol": "in_place_ttt_2x5090_ruler_matrix_audit_v1",
                "passed": True,
                "errors": [],
                "recomputed_context_scores": {
                    suffix: {} for suffix in (
                        "4k", "8k", "16k", "32k", "64k", "128k"
                    )
                },
            })
            public_plan = Path(temporary) / "public.md"
            evaluation_plan = Path(temporary) / "evaluation.md"
            extrapolation_plan = Path(temporary) / "extrapolation.md"
            runner = Path(temporary) / "runner.sh"
            for path in (public_plan, evaluation_plan, extrapolation_plan, runner):
                path.write_text(path.name + "\n")

            root.mkdir()
            (root / "state.txt").write_text("auditing_256k_resource\n")
            for arm in ("no_ttt", "ttt"):
                child = root / arm
                predictions = child / "opencompass/date/predictions/model"
                for prefix in TASK_PREFIXES:
                    write_json(predictions / f"{prefix}_256k.json", {
                        "0": {
                            "origin_prompt": "shared prompt",
                            "references": "shared answer",
                            "prediction": arm,
                        }
                    })
                write_json(child / "run_metadata.json", {
                    "protocol": "in_place_ttt_2x5090_ruler_run_v1",
                    "run_kind": "resource",
                    "num_samples_per_task": 1,
                    "context_length": 262144,
                    "extrapolation": True,
                    "arm": arm,
                })
                write_json(child / "independent_resource_audit.json", {
                    "protocol": "in_place_ttt_2x5090_ruler_resource_audit_v1",
                    "passed": True,
                    "errors": [],
                    "arm": arm,
                    "context_length": 262144,
                    "extrapolation": True,
                    "gpu": {
                        "peak_memory_used_mib": 20000,
                        "memory_total_mib": 32607,
                    },
                })

            tool = Path(__file__).resolve().parents[1]
            auditor = tool / "scripts/audit_in_place_ttt_2x5090_ruler_256k_resource.py"
            single = tool / "scripts/run_in_place_ttt_2x5090_ruler_single.sh"
            write_json(root / "resource_metadata.json", {
                "protocol": "in_place_ttt_2x5090_ruler_256k_resource_run_v1",
                "hostname": "user-Super-Server",
                "tool_clean": True,
                "tool_commit": "a" * 40,
                "training_root": str(training.resolve()),
                "matrix_root": str(matrix.resolve()),
                "context_length": 262144,
                "training_audit_sha256": sha(training_audit),
                "matrix_audit_sha256": sha(matrix_audit),
                "public_corpus_plan_sha256": sha(public_plan),
                "evaluation_plan_sha256": sha(evaluation_plan),
                "extrapolation_plan_sha256": sha(extrapolation_plan),
                "runner_sha256": sha(runner),
                "single_runner_sha256": sha(single),
                "auditor_sha256": sha(auditor),
            })
            kwargs = {
                "training_root": training,
                "matrix_root": matrix,
                "public_corpus_plan": public_plan,
                "evaluation_plan": evaluation_plan,
                "extrapolation_plan": extrapolation_plan,
                "runner": runner,
            }
            passed = audit(root, **kwargs)
            self.assertTrue(passed["passed"], passed["errors"])
            self.assertTrue(passed["formal_256k_released"])

            changed = next(
                (root / "ttt/opencompass/date/predictions/model").glob("*.json")
            )
            payload = json.loads(changed.read_text())
            payload["0"]["origin_prompt"] = "different"
            write_json(changed, payload)
            failed = audit(root, **kwargs)
            self.assertFalse(failed["passed"])
            self.assertIn("paired 256K resource inputs differ", failed["errors"])


class TwoRTX5090Ruler256KFormalTest(unittest.TestCase):
    def test_release_requires_explicit_acceptance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            resource = root / "resource.json"
            output = root / "release.json"
            write_json(resource, {
                "protocol": "in_place_ttt_2x5090_ruler_256k_resource_audit_v1",
                "passed": True,
                "errors": [],
                "context_length": 262144,
                "extrapolation": True,
                "scores_discarded": True,
                "formal_256k_released": True,
                "boundaries": {"no_ttt": {}, "ttt": {}},
            })
            with self.assertRaises(PermissionError):
                create_release(
                    resource_audit=resource, output=output, accept=False,
                    reviewed_by="reviewer", decision_note="resource passed",
                )
            self.assertFalse(output.exists())
            release = create_release(
                resource_audit=resource, output=output, accept=True,
                reviewed_by="reviewer", decision_note="resource passed",
            )
            self.assertTrue(release["accepted"])
            self.assertEqual(release["resource_audit_sha256"], sha(resource))

    def test_formal_summary_and_independent_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "formal"
            for arm, score in (("no_ttt", 40.0), ("ttt", 43.0)):
                child = root / arm
                result_dir = child / "opencompass/date/results/model"
                prediction_dir = child / "opencompass/date/predictions/model"
                task_scores = {}
                input_hashes = {}
                for prefix in TASK_PREFIXES:
                    task = f"{prefix}_256k"
                    payload = {
                        str(index): {
                            "origin_prompt": f"prompt {index}",
                            "references": f"answer {index}",
                            "prediction": arm,
                        }
                        for index in range(100)
                    }
                    write_json(result_dir / f"{task}.json", {"score": score})
                    write_json(prediction_dir / f"{task}.json", payload)
                    task_scores[task] = score
                    input_hashes[task] = evaluation_input_sha256(payload)
                write_json(child / "run_metadata.json", {
                    "run_kind": "formal",
                    "num_samples_per_task": 100,
                    "arm": arm,
                    "context_length": 262144,
                    "extrapolation": True,
                })
                write_json(child / "summary.json", {
                    "protocol": "in_place_ttt_2x5090_ruler_summary_v1",
                    "completed": True,
                    "arm": arm,
                    "context_length": 262144,
                    "macro_accuracy": score,
                    "task_scores": task_scores,
                    "evaluation_input_sha256": input_hashes,
                })
                write_json(child / "independent_audit.json", {
                    "protocol": "in_place_ttt_2x5090_ruler_audit_v1",
                    "passed": True,
                    "errors": [],
                    "arm": arm,
                    "context_length": 262144,
                    "extrapolation": True,
                })
            summary = summarize(root)
            self.assertTrue(summary["completed"], summary["errors"])
            self.assertEqual(summary["scores"]["ttt_minus_no_ttt"], 3.0)
            write_json(root / "summary.json", summary)
            (root / "state.txt").write_text("auditing_ruler_256k\n")

            training = base / "training"
            matrix = base / "matrix"
            resource = base / "resource"
            for path in (training, matrix, resource):
                path.mkdir()
            training_audit = training / "independent_audit.json"
            matrix_audit = matrix / "independent_matrix_audit.json"
            resource_audit = resource / "independent_resource_audit.json"
            write_json(training_audit, {
                "protocol": "in_place_ttt_2x5090_100m_stage2_audit_v1",
                "passed": True, "errors": [],
                "arms": {"no_ttt": {}, "ttt": {}},
            })
            write_json(matrix_audit, {
                "protocol": "in_place_ttt_2x5090_ruler_matrix_audit_v1",
                "passed": True, "errors": [],
            })
            write_json(resource_audit, {
                "protocol": "in_place_ttt_2x5090_ruler_256k_resource_audit_v1",
                "passed": True, "errors": [], "formal_256k_released": True,
            })
            release = base / "release.json"
            write_json(release, {
                "protocol": "in_place_ttt_2x5090_ruler_256k_formal_release_v1",
                "accepted": True,
                "resource_audit_sha256": sha(resource_audit),
                "reviewed_by": "reviewer",
                "decision_note": "resource passed",
            })
            plans = {
                name: base / f"{name}.md" for name in (
                    "public", "evaluation", "extrapolation", "formal"
                )
            }
            for name, path in plans.items():
                path.write_text(name + "\n")
            runner = base / "runner.sh"
            runner.write_text("runner\n")
            tool = Path(__file__).resolve().parents[1]
            write_json(root / "formal_metadata.json", {
                "protocol": "in_place_ttt_2x5090_ruler_256k_formal_run_v1",
                "hostname": "user-Super-Server",
                "tool_clean": True,
                "tool_commit": "a" * 40,
                "context_length": 262144,
                "training_root": str(training.resolve()),
                "matrix_root": str(matrix.resolve()),
                "resource_root": str(resource.resolve()),
                "training_audit_sha256": sha(training_audit),
                "matrix_audit_sha256": sha(matrix_audit),
                "resource_audit_sha256": sha(resource_audit),
                "release_sha256": sha(release),
                "public_corpus_plan_sha256": sha(plans["public"]),
                "evaluation_plan_sha256": sha(plans["evaluation"]),
                "extrapolation_plan_sha256": sha(plans["extrapolation"]),
                "formal_plan_sha256": sha(plans["formal"]),
                "runner_sha256": sha(runner),
                "single_runner_sha256": sha(
                    tool / "scripts/run_in_place_ttt_2x5090_ruler_single.sh"
                ),
                "summarizer_sha256": sha(
                    tool / "scripts/summarize_in_place_ttt_2x5090_ruler_256k.py"
                ),
                "auditor_sha256": sha(
                    tool / "scripts/audit_in_place_ttt_2x5090_ruler_256k.py"
                ),
            })
            kwargs = {
                "training_root": training,
                "matrix_root": matrix,
                "resource_root": resource,
                "release": release,
                "public_corpus_plan": plans["public"],
                "evaluation_plan": plans["evaluation"],
                "extrapolation_plan": plans["extrapolation"],
                "formal_plan": plans["formal"],
                "runner": runner,
            }
            checked = audit_formal(root, **kwargs)
            self.assertTrue(checked["passed"], checked["errors"])
            self.assertEqual(checked["scores"]["ttt_minus_no_ttt"], 3.0)

            raw = next((root / "ttt/opencompass/date/results/model").glob("*.json"))
            write_json(raw, {"score": 44.0})
            failed = audit_formal(root, **kwargs)
            self.assertFalse(failed["passed"])
            self.assertIn(
                "formal 256K summary does not reproduce raw artifacts",
                failed["errors"],
            )


if __name__ == "__main__":
    unittest.main()
