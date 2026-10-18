import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.summarize_in_place_ttt_baseline_resource import summarize


class BaselineResourceSummaryTest(unittest.TestCase):
    def test_completed_probe_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data.jsonl"; data.write_text("{}\n")
            config = root / "config.yaml"; config.write_text("train: {}\n")
            plan = root / "plan.md"; plan.write_text("frozen\n")
            model_config = root / "model_config.json"; model_config.write_text("{}\n")
            sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            (root / "run_metadata.json").write_text(json.dumps({
                "upstream_commit": "be2324829b0e91c8fd10a74d4b43714fde6676e1",
                "arm": "ttt", "sequence_length": 8192, "global_batch_size": 2,
                "protocol": "in_place_ttt_baseline_resource_v2",
                "resource_profile": "cpu_offload",
                "plan_sha256": sha(plan), "config_sha256": sha(config),
                "data_sha256": sha(data), "model_config_sha256": sha(model_config),
            }))
            (root / "exit_code.txt").write_text("0\n")
            (root / "train.log").write_text(
                "{'loss': 2.5, 'grad_norm': 0.75}\n"
                "Elapsed (wall clock) time (h:mm:ss or m:ss): 1:00.00\n"
            )
            with (root / "gpu.csv").open("w", newline="") as sink:
                writer = csv.DictWriter(sink, fieldnames=(
                    "timestamp", "index", "memory_used_mib", "memory_total_mib",
                    "utilization_gpu"))
                writer.writeheader()
                writer.writerow({"timestamp": 1, "index": 0, "memory_used_mib": 100,
                                 "memory_total_mib": 1000, "utilization_gpu": 50})
                writer.writerow({"timestamp": 2, "index": 0, "memory_used_mib": 200,
                                 "memory_total_mib": 1000, "utilization_gpu": 80})
            with (root / "host.csv").open("w", newline="") as sink:
                writer = csv.DictWriter(sink, fieldnames=(
                    "timestamp", "memory_total_mib", "memory_used_mib",
                    "memory_available_mib", "swap_total_mib", "swap_used_mib"))
                writer.writeheader()
                writer.writerow({"timestamp": 1, "memory_total_mib": 64000,
                                 "memory_used_mib": 12000,
                                 "memory_available_mib": 52000,
                                 "swap_total_mib": 4096, "swap_used_mib": 512})
            result = summarize(root, root / "summary.json")
            self.assertTrue(result["completed"])
            self.assertEqual(result["last_loss"], 2.5)
            self.assertEqual(result["last_gradient_norm"], .75)
            self.assertEqual(result["wall_seconds"], 60)
            self.assertAlmostEqual(result["tokens_per_second"], 16384 / 60)
            self.assertEqual(result["peak_gpu_memory_mib"], {"0": 200})
            self.assertEqual(result["minimum_gpu_free_fraction"], {"0": .8})
            self.assertTrue(result["cpu_offload_host_gate_passed"])
            self.assertEqual(result["host_resources"]["peak_swap_used_mib"], 512)

    def test_activation_offload_gate_and_step_throughput(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, content in (
                ("data.jsonl", "{}\n"), ("config.yaml", "train: {}\n"),
                ("plan.md", "frozen\n"), ("model_config.json", "{}\n"),
            ):
                (root / name).write_text(content)
            sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            (root / "run_metadata.json").write_text(json.dumps({
                "upstream_commit": "be2324829b0e91c8fd10a74d4b43714fde6676e1",
                "arm": "no_ttt", "sequence_length": 32768, "global_batch_size": 2,
                "protocol": "in_place_ttt_baseline_resource_v2",
                "resource_profile": "activation_offload",
                "plan_sha256": sha(root / "plan.md"),
                "config_sha256": sha(root / "config.yaml"),
                "data_sha256": sha(root / "data.jsonl"),
                "model_config_sha256": sha(root / "model_config.json"),
            }))
            (root / "exit_code.txt").write_text("0\n")
            (root / "train.log").write_text(
                "Epoch: 100%|x| 1/1 [00:10<00:00, 10.00s/it, loss: 2.5, "
                "grad_norm: 0.75, lr: 1e-7]\n"
                "Elapsed (wall clock) time (h:mm:ss or m:ss): 0:30.00\n"
            )
            with (root / "gpu.csv").open("w", newline="") as sink:
                writer = csv.DictWriter(sink, fieldnames=(
                    "timestamp", "index", "memory_used_mib", "memory_total_mib",
                    "utilization_gpu"))
                writer.writeheader()
                for index in (0, 1):
                    writer.writerow({"timestamp": 1, "index": index,
                                     "memory_used_mib": 30000,
                                     "memory_total_mib": 32607,
                                     "utilization_gpu": 100})
            with (root / "host.csv").open("w", newline="") as sink:
                writer = csv.DictWriter(sink, fieldnames=(
                    "timestamp", "memory_total_mib", "memory_used_mib",
                    "memory_available_mib", "swap_total_mib", "swap_used_mib"))
                writer.writeheader()
                writer.writerow({"timestamp": 1, "memory_total_mib": 64000,
                                 "memory_used_mib": 48000,
                                 "memory_available_mib": 16000,
                                 "swap_total_mib": 4096, "swap_used_mib": 512})
            result = summarize(root, root / "summary.json")
            self.assertTrue(result["activation_offload_gate_passed"])
            self.assertEqual(result["optimizer_step_seconds"], 10)
            self.assertEqual(result["steady_tokens_per_second"], 6553.6)


if __name__ == "__main__":
    unittest.main()
