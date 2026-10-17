import csv
import tempfile
import unittest
from pathlib import Path

from scripts.audit_in_place_ttt_level_b_preflight import parse_gpu_trace, parse_metrics


class LevelBPreflightAuditTest(unittest.TestCase):
    def test_metric_parser_uses_last_finite_step(self):
        log = "loss: 2.3, grad_norm: 4.5, lr: 5.00e-06\rprogress"
        self.assertEqual(parse_metrics(log), {"loss": 2.3, "grad_norm": 4.5, "lr": 5e-6})

    def test_gpu_trace_requires_every_selected_device(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "gpu.csv"
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["timestamp", "index", "memory_used_mib",
                                 "memory_total_mib", "utilization_gpu"])
                for index in range(8):
                    writer.writerow([1.0, index, 76000, 81920, 99])
            result = parse_gpu_trace(path, list(range(8)))
            self.assertTrue(result["complete"])
            self.assertGreater(result["minimum_free_fraction"], 0.05)
            incomplete = parse_gpu_trace(path, list(range(9)))
            self.assertFalse(incomplete["complete"])


if __name__ == "__main__":
    unittest.main()
