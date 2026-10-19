import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.monitor_in_place_ttt_public_corpus_memory import (
    BUILDER_NAME,
    collect_sample,
    matching_builder_pids,
)


class PublicCorpusMemoryMonitorTest(unittest.TestCase):
    def test_builder_match_requires_exact_output_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proc = root / "proc"
            target = root / "corpus"
            for pid, output in (("11", target), ("12", root / "other")):
                process = proc / pid
                process.mkdir(parents=True)
                (process / "cmdline").write_bytes(
                    b"python\0" + BUILDER_NAME.encode() + b"\0--output-root\0"
                    + str(output).encode() + b"\0"
                )
            self.assertEqual(matching_builder_pids(proc, target), [11])

    @mock.patch("scripts.monitor_in_place_ttt_public_corpus_memory.shutil.disk_usage")
    def test_sample_uses_only_scalar_process_and_file_metadata(self, disk_usage):
        disk_usage.return_value = shutil_usage(total=1000, used=400, free=600)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proc = root / "proc" / "11"
            proc.mkdir(parents=True)
            (proc / "status").write_text(
                "VmRSS:\t1234 kB\nRssAnon:\t1000 kB\nVmSwap:\t7 kB\nThreads:\t9\n"
            )
            meminfo = root / "meminfo"
            meminfo.write_text(
                "MemTotal: 10000 kB\nMemAvailable: 8000 kB\n"
                "SwapTotal: 1000 kB\nSwapFree: 900 kB\n"
            )
            partial = root / "corpus.partial"
            partial.mkdir()
            (partial / "stage1_32k.jsonl").write_bytes(b"12345")
            status = root / "status.txt"
            status.write_text("RUNNING_BUILD\n")
            args = argparse.Namespace(
                proc_root=root / "proc",
                meminfo=meminfo,
                partial_root=partial,
                wrapper_status=status,
            )
            row = collect_sample(args, 11)
            self.assertEqual(row["vm_rss_kib"], 1234)
            self.assertEqual(row["rss_anon_kib"], 1000)
            self.assertEqual(row["vm_swap_kib"], 7)
            self.assertEqual(row["host_mem_available_kib"], 8000)
            self.assertEqual(row["host_swap_used_kib"], 100)
            self.assertEqual(row["stage1_data_bytes"], 5)
            self.assertEqual(row["stage2_data_bytes"], 0)
            self.assertEqual(row["wrapper_status"], "RUNNING_BUILD")

    def test_plan_and_source_are_read_only(self):
        root = Path(__file__).resolve().parents[1]
        plan = (root / "IN_PLACE_TTT_BASELINE_PUBLIC_CORPUS_MEMORY_MONITOR_ADDENDUM.md").read_text()
        source = (root / "scripts/monitor_in_place_ttt_public_corpus_memory.py").read_text()
        self.assertIn("frozen before persistent monitoring", plan)
        self.assertIn("must not scan JSONL records", plan)
        self.assertNotIn("os.kill", source)
        self.assertNotIn("send_signal", source)
        self.assertNotIn("terminate(", source)
        self.assertNotIn("read_text(encoding=\"utf-8\")\n        for name in PARTIAL_FILES", source)


def shutil_usage(total, used, free):
    return type("usage", (), {"total": total, "used": used, "free": free})()


if __name__ == "__main__":
    unittest.main()
