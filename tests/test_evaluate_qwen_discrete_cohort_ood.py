import json
from pathlib import Path
import tempfile
import unittest

from tasks.pack_cbf_event_warmup import sha
from tasks.evaluate_qwen_discrete_cohort_ood import (
    CONFIRMATION_AUDIT_PROTOCOL,
    CONFIRMATION_PROTOCOL,
    SUITES,
    registered_gate,
    verify_confirmation,
)
from tasks.train_cbf_qwen_lowrank_forgetting import SEEDS
from scripts.audit_qwen_discrete_cohort_ood import (
    action_summary as audit_action_summary,
    gate as audit_gate,
)


class QwenDiscreteCohortOodEvaluationTest(unittest.TestCase):
    @staticmethod
    def policy(accuracy, nll, historical=0.97, endpoint=0.99, interior=0.7):
        del endpoint, interior
        return {
            "mean_accuracy": accuracy,
            "mean_nll": nll,
            "min_seed_accuracy": accuracy,
            "roles": {"current": 1.0, "historical": historical, "anchor": 1.0},
            "per_seed": {
                str(seed): {
                    "accuracy": accuracy,
                    "groups": {
                        f"g{index}": {"accuracy": accuracy, "nll": nll}
                        for index in range(4)
                    },
                }
                for seed in SEEDS
            },
        }

    def passing(self):
        macro = {
            "grid_classifier": self.policy(0.98, 0.15),
            "physics_cohort": self.policy(0.94, 0.35, historical=0.8),
            "oracle_grid": self.policy(0.99, 0.14),
        }
        suites = {
            suite: {
                "grid_classifier": self.policy(0.98, 0.15),
                "physics_cohort": self.policy(0.94, 0.35),
            }
            for suite in SUITES
        }
        action = {
            "buckets": {
                "endpoint": {"action_accuracy": 0.99},
                "interior": {"action_accuracy": 0.70},
            }
        }
        return suites, macro, action

    def test_registered_gate(self):
        suites, macro, action = self.passing()
        result = registered_gate(suites, macro, action)
        self.assertTrue(result["passed"])
        self.assertEqual(result, audit_gate(suites, macro, action))
        suites["long_session"]["grid_classifier"] = self.policy(0.94, 0.2)
        self.assertFalse(registered_gate(suites, macro, action)["passed"])

    def test_independent_action_audit_separates_grid_interior(self):
        records = {
            str(seed): [
                {
                    "queries": [{"correct": 1}] * 6,
                    "decisions": [
                        {
                            "oracle_retention": [0.0, 1.0, 1.0],
                            "retention": [0.0, 1.0, 1.0],
                            "retention_mae": 0.0,
                        },
                        {
                            "oracle_retention": [0.5, 1.0, 1.0],
                            "retention": [0.0, 1.0, 1.0],
                            "retention_mae": 1.0 / 6.0,
                        },
                    ],
                }
            ]
            for seed in SEEDS
        }
        summary = audit_action_summary(records)
        self.assertEqual(summary["decisions"], 6)
        self.assertEqual(summary["buckets"]["endpoint"]["action_accuracy"], 1.0)
        self.assertEqual(summary["buckets"]["interior"]["action_accuracy"], 0.0)
        self.assertEqual(summary["oracle_interior_fraction"], 0.5)

    def test_confirmation_dependency_is_hash_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = root / "result"
            result.mkdir()
            summary = {
                "protocol": CONFIRMATION_PROTOCOL,
                "test_scored": True,
                "stage_passed": True,
                "test_gate": {"passed": True},
            }
            (result / "summary.json").write_text(json.dumps(summary))
            (result / "controllers.pt").write_bytes(b"checkpoint")
            audit = {
                "protocol": CONFIRMATION_AUDIT_PROTOCOL,
                "audit_passed": True,
                "test_scored": True,
                "test_gate": {"passed": True},
                "summary_sha256": sha(result / "summary.json"),
                "checkpoint_sha256": sha(result / "controllers.pt"),
            }
            audit_path = root / "execution_audit.json"
            audit_path.write_text(json.dumps(audit))
            verified = verify_confirmation(root, audit_path)
            self.assertEqual(verified["confirmation_summary_sha256"],
                             sha(result / "summary.json"))
            summary["stage_passed"] = False
            (result / "summary.json").write_text(json.dumps(summary))
            with self.assertRaises(ValueError):
                verify_confirmation(root, audit_path)


if __name__ == "__main__":
    unittest.main()
