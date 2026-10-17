import math
import json
import csv
import tempfile
import unittest
from pathlib import Path

from scripts.audit_in_place_ttt_level_b_training import (
    all_metrics,
    audit,
    expected_checkpoint_steps,
    sha256,
)


class LevelBTrainingAuditTest(unittest.TestCase):
    def test_registered_checkpoint_steps(self):
        stage1 = expected_checkpoint_steps(9537)
        stage2 = expected_checkpoint_steps(1788)
        self.assertEqual(stage1[:2], [500, 1000])
        self.assertEqual(stage1[-2:], [9500, 9537])
        self.assertEqual(len(stage1), 20)
        self.assertEqual(stage2, [500, 1000, 1500, 1788])

    def test_metric_parser_retains_all_observed_metrics(self):
        rows = all_metrics(
            "loss: 2.0, grad_norm: 3.0, lr: 5e-6\r"
            "loss: 1.5, grad_norm: 2.5, lr: 4e-6"
        )
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(math.isfinite(value) for row in rows for value in row.values()))
        self.assertEqual(rows[-1]["loss"], 1.5)

    def test_complete_stage2_fixture_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            addendum = root / "addendum.md"
            addendum.write_text("frozen\n", encoding="utf-8")
            config_audit = root / "config_audit.json"
            config_audit.write_text(json.dumps({"passed": True, "corpus": {"root": "x"}}))
            preflight = root / "preflight.json"
            preflight.write_text(json.dumps({
                "passed": True, "stage": "stage2",
                "arms": {"no_ttt": {"passed": True}},
            }))
            preflight_metadata = root / "preflight_metadata.json"
            preflight_metadata.write_text(json.dumps({
                "hostname": "fixture", "stage": "stage2",
                "selected_gpus": list(range(8)),
            }))
            metadata = {
                "protocol": "in_place_ttt_level_b_training_v1",
                "stage": "stage2", "arm": "no_ttt",
                "expected_steps": 1788, "seq_len": 131072,
                "train_size": 14_998_831_104,
                "selected_gpus": list(range(8)),
                "hostname": "fixture",
                "upstream_commit": "be2324829b0e91c8fd10a74d4b43714fde6676e1",
                "upstream_clean": True,
                "config_audit_sha256": sha256(config_audit),
                "training_addendum_sha256": sha256(addendum),
                "preflight_audit_path": str(preflight),
                "preflight_audit_sha256": sha256(preflight),
                "preflight_run_metadata_path": str(preflight_metadata),
                "preflight_run_metadata_sha256": sha256(preflight_metadata),
            }
            (root / "run_metadata.json").write_text(json.dumps(metadata))
            (root / "train.log").write_text(
                "loss: 2.0, grad_norm: 3.0, lr: 5e-6\n", encoding="utf-8")
            (root / "exit_code.txt").write_text("0\n", encoding="utf-8")
            with (root / "gpu.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["timestamp", "index", "memory_used_mib",
                                 "memory_total_mib", "utilization_gpu"])
                for index in range(8):
                    writer.writerow([1, index, 76000, 81920, 90])
            with (root / "host.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["timestamp", "memory_used_bytes",
                                 "memory_available_bytes", "swap_used_bytes"])
                writer.writerow([1, 100, 200, 0])
            checkpoint_root = root / "training_output" / "checkpoint" / "checkpoints"
            for step in expected_checkpoint_steps(1788):
                checkpoint = checkpoint_root / f"global_step_{step}"
                checkpoint.mkdir(parents=True)
                (checkpoint / ".metadata").write_text("metadata")
                with (checkpoint / "rank.distcp").open("wb") as handle:
                    handle.seek(40_000_000_000 - 1); handle.write(b"0")
            hf = checkpoint_root / "global_step_1788" / "hf_ckpt"
            hf.mkdir()
            (hf / "config.json").write_text(json.dumps({
                "ttt_mode": False, "max_position_embeddings": 131072,
            }))
            (hf / "model.safetensors.index.json").write_text("{}")
            published = root.parent / "published"
            result = audit(root, "stage2", "no_ttt", addendum, published)
            self.assertTrue(result["passed"], result["errors"])
            self.assertEqual(
                Path(result["final_hf_path"]),
                published / result["final_hf_relative_path"],
            )


if __name__ == "__main__":
    unittest.main()
