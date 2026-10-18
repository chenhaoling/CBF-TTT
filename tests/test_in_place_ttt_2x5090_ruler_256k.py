from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_in_place_ttt_2x5090_ruler_256k_resource import audit
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


if __name__ == "__main__":
    unittest.main()
