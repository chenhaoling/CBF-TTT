import unittest
from pathlib import Path
import json
import tempfile

from tasks.build_native_support_write_gate_v1 import sha256
from tasks.evaluate_native_eos_safe_layerwise_write_gate_v4 import release_checks
from tasks.native_eos_safe_layerwise_write_gate_v4 import verify_stage_release
from tasks.train_native_eos_safe_layerwise_write_gate_v4 import (
    build_model,
    choose_actions,
    feature_indices,
    predict,
)


class NativeEosSafeLayerwiseControllerV4Test(unittest.TestCase):
    @staticmethod
    def _write(path: Path, value: dict) -> None:
        path.write_text(json.dumps(value) + "\n")

    def test_feature_ablations_are_disjoint_and_complete(self):
        no_support = feature_indices("no_support")
        support_only = feature_indices("support_only")
        full = feature_indices("full")
        self.assertEqual(len(full), 1854)
        self.assertEqual(len(no_support), 41)
        self.assertEqual(len(support_only), 1815)
        self.assertEqual(set(no_support) | set(support_only), set(full))
        self.assertEqual(set(no_support) & set(support_only), {1852, 1853})

    def test_four_head_output_and_eos_veto(self):
        import torch

        model = build_model(41)
        x = torch.zeros(2, 41)
        stats = {
            "x_mean": torch.zeros(41), "x_std": torch.ones(41),
            "benefit_mean": torch.zeros(7), "benefit_std": torch.ones(7),
            "delta_mean": torch.zeros(7), "delta_std": torch.ones(7),
        }
        outputs = predict(model, x, stats)
        self.assertEqual(len(outputs), 4)
        self.assertTrue(all(value.shape == (2, 7) for value in outputs))
        benefit = torch.ones(1, 7)
        delta = torch.zeros(1, 7)
        safe = torch.zeros(1, 7)
        eos_harm = torch.ones(1, 7)
        self.assertEqual(choose_actions(benefit, delta, safe, eos_harm), ["reject_all"])

    def test_sealed_test_requires_audited_development_release(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); controller = root / "controller"; controller.mkdir()
            train_summary = root / "train_summary.json"
            self._write(train_summary, {"passed": True})
            self._write(root / "train_independent_audit.json", {
                "passed": True, "test_opened": False, "summary_sha256": sha256(train_summary),
            })
            dev_summary = root / "development_summary.json"
            self._write(dev_summary, {"passed": True})
            self._write(root / "development_independent_audit.json", {
                "passed": True, "test_opened": False, "summary_sha256": sha256(dev_summary),
            })
            controller_summary = controller / "training_summary.json"
            self._write(controller_summary, {
                "development_released_test": True, "test_opened": False,
            })
            controller_audit = controller / "training_independent_audit.json"
            self._write(controller_audit, {
                "passed": True, "test_released": True, "test_opened": False,
                "summary_sha256": sha256(controller_summary),
            })
            self._write(controller / "sealed_test_release.json", {
                "artifact_root": str(root.resolve()),
                "controller_summary_sha256": sha256(controller_summary),
                "controller_independent_audit_sha256": sha256(controller_audit),
            })
            dependency = verify_stage_release("sealed_test", root, controller)
            self.assertEqual(dependency["controller_summary_sha256"], sha256(controller_summary))
            self._write(controller_audit, {
                "passed": True, "test_released": False, "test_opened": False,
                "summary_sha256": sha256(controller_summary),
            })
            with self.assertRaises(ValueError):
                verify_stage_release("sealed_test", root, controller)

    def test_registered_release_checks_include_support_ablation_and_safety(self):
        never = {"J": 1.0, "protected_accuracy": .8, "protected_nll": .5, "eos_rate": .1}
        always = {"J": 1.2, "protected_accuracy": .7, "protected_nll": .7, "eos_rate": .2}
        oracle = {"J": .8, "protected_accuracy": .8, "protected_nll": .5, "eos_rate": .1}
        metrics = {"J": .9, "protected_accuracy": .8, "protected_nll": .5, "eos_rate": .1}
        full = {
            "test_vs_always_write": {"ci95": [.1, .3]},
            "test_vs_never_write": {"ci95": [.02, .18]},
        }
        no_support = {"test_metrics": {"J": .95}}
        result = release_checks(metrics, full, no_support, {
            "never_write": never, "always_write": always, "safe_oracle": oracle,
        })
        self.assertTrue(result["passed"])
        no_support["test_metrics"]["J"] = .85
        result = release_checks(metrics, full, no_support, {
            "never_write": never, "always_write": always, "safe_oracle": oracle,
        })
        self.assertFalse(result["checks"]["full_better_than_no_support"])
        self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
