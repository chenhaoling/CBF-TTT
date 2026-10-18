import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_in_place_ttt_2x5090_resource import audit
from scripts.audit_in_place_ttt_level_b_configs import sha256


class TwoRTX5090ResourceAuditTest(unittest.TestCase):
    def fixture(self, root: Path) -> dict[str, Path]:
        paths = {}
        contents = {
            "plan": "parent plan\n", "addendum": "frozen 2x5090 plan\n",
            "data": '{"content_split": [1]}\n', "model_config": "{}\n",
            "runner": "#!/bin/bash\n", "summarizer": "# summarizer\n",
            "offloading": "# activation offloading\n",
        }
        for key, content in contents.items():
            paths[key] = root / f"{key}.txt"
            paths[key].write_text(content)
        paths["config"] = root / "config.yaml"
        paths["config"].write_text(
            "model:\n  foundation:\n    ttt_mode: false\n"
            "train:\n  data_parallel_mode: fsdp2\n"
            "  enable_gradient_checkpointing: true\n"
            "  enable_activation_offload: true\n"
            "  activation_gpu_limit: 0.0\n"
            "  enable_fsdp_offload: false\n"
        )
        (root / "preflight_gpu.csv").write_text(
            "0, NVIDIA GeForce RTX 5090, 32607, 46\n"
            "1, NVIDIA GeForce RTX 5090, 32607, 18\n"
        )
        (root / "preflight_processes.csv").write_text("")
        (root / "preflight_conflicts.txt").write_text("")
        (root / "preflight_host.txt").write_text(
            "MemTotal:       65000000 kB\nMemAvailable:   52000000 kB\n"
        )
        metadata = {
            "protocol": "in_place_ttt_baseline_resource_v2",
            "resource_profile": "activation_offload", "activation_offload": True,
            "hostname": "user-Super-Server", "upstream_commit":
                "be2324829b0e91c8fd10a74d4b43714fde6676e1",
            "upstream_clean": True, "arm": "no_ttt", "sequence_length": 8192,
            "global_batch_size": 2,
            "pytorch_cuda_alloc_conf": "expandable_segments:True",
        }
        bindings = {
            "plan_sha256": paths["plan"], "addendum_sha256": paths["addendum"],
            "config_sha256": paths["config"], "data_sha256": paths["data"],
            "model_config_sha256": paths["model_config"],
            "runner_sha256": paths["runner"],
            "summarizer_sha256": paths["summarizer"],
            "activation_offloading_sha256": paths["offloading"],
            "preflight_gpu_sha256": root / "preflight_gpu.csv",
            "preflight_processes_sha256": root / "preflight_processes.csv",
            "preflight_host_sha256": root / "preflight_host.txt",
            "preflight_conflicts_sha256": root / "preflight_conflicts.txt",
        }
        metadata.update({key: sha256(path) for key, path in bindings.items()})
        (root / "run_metadata.json").write_text(json.dumps(metadata))
        log = (
            '"data_parallel_mode": "fsdp2"\n'
            '"enable_activation_offload": true\n"activation_gpu_limit": 0.0\n'
            '"enable_fsdp_offload": false\nloss: 2.5, grad_norm: 0.7\n'
        )
        (root / "train.log").write_text(log)
        (root / "exit_code.txt").write_text("0\n")
        with (root / "gpu.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["timestamp", "index", "memory_used_mib",
                             "memory_total_mib", "utilization_gpu"])
            writer.writerow([1, 0, 30000, 32607, 99])
            writer.writerow([1, 1, 29900, 32607, 99])
        with (root / "host.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["timestamp", "memory_total_mib", "memory_used_mib",
                             "memory_available_mib", "swap_total_mib", "swap_used_mib"])
            writer.writerow([1, 62000, 46000, 16000, 4096, 512])
        checkpoint = root / "checkpoint" / "global_step_1"
        checkpoint.mkdir(parents=True)
        (checkpoint / ".metadata").write_text("metadata")
        with (checkpoint / "rank.distcp").open("wb") as handle:
            handle.seek(40_000_000_000 - 1)
            handle.write(b"0")
        checkpoint_bytes = sum(
            path.stat().st_size for path in (root / "checkpoint").rglob("*")
            if path.is_file()
        )
        dependencies = {key: sha256(path) for key, path in bindings.items()}
        dependencies.update({
            "train_log_sha256": sha256(root / "train.log"),
            "gpu_monitor_sha256": sha256(root / "gpu.csv"),
            "host_monitor_sha256": sha256(root / "host.csv"),
        })
        summary = {
            "dependencies": dependencies, "exit_code": 0, "completed": True,
            "activation_offload_gate_passed": True, "last_loss": 2.5,
            "last_gradient_norm": 0.7, "optimizer_step_seconds": 10.0,
            "minimum_gpu_free_fraction": {"0": 0.0799, "1": 0.083},
            "host_resources": {"minimum_memory_available_mib": 16000,
                               "peak_swap_used_mib": 512},
            "checkpoint_bytes": checkpoint_bytes,
        }
        (root / "summary.json").write_text(json.dumps(summary))
        return paths

    def test_complete_fixture_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self.fixture(root)
            result = audit(root, **paths)
            self.assertTrue(result["passed"], result["errors"])
            self.assertEqual([row["index"] for row in result["preflight_gpus"]], [0, 1])

    def test_active_corpus_builder_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self.fixture(root)
            (root / "preflight_conflicts.txt").write_text("123 builder.py\n")
            result = audit(root, **paths)
            self.assertFalse(result["passed"])
            self.assertTrue(any("conflict list" in error for error in result["errors"]))


if __name__ == "__main__":
    unittest.main()
