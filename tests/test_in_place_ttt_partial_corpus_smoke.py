import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_in_place_ttt_partial_corpus_smoke import (
    SEQ_LEN,
    first_complete_lines,
    last_complete_lines,
    provenance_errors,
    record_errors,
    token_sha,
)


class FakeTokenizer:
    eos_token_id = 151643

    def encode(self, text, add_special_tokens=False):
        self.assert_special = add_special_tokens
        return list(range(SEQ_LEN - 1)) if text == "fixture" else []


def provenance(index):
    source = ("fineweb", "longcrawl")[index % 2]
    ids = list(range(SEQ_LEN - 1)) + [151643]
    return {
        "source": source,
        "stage": "stage1_32k",
        "stage_record_index": index,
        "source_record_index": index // 2,
        "content_token_count": SEQ_LEN - 1,
        "training_token_count": SEQ_LEN,
        "raw_source_tokens_consumed": SEQ_LEN - 1,
        "synthetic_eos_tokens": 0,
        "training_token_sha256": token_sha(ids),
        "spans": [{
            "source": source,
            "document_index": index,
            "locator": {"file_index": 0, "repository_path": "part.parquet",
                        "parquet_row_index": index},
            "document_sha256": "a" * 64,
            "content_token_count": SEQ_LEN - 1,
            "token_count_with_boundary": SEQ_LEN,
            "token_start": 0,
            "token_end": SEQ_LEN - 1,
        }],
    }


class PartialCorpusSmokeTest(unittest.TestCase):
    def test_first_and_tail_read_only_complete_lines(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rows.jsonl"
            path.write_bytes(b'{"i":0}\n{"i":1}\n{"i":2}\npartial')
            self.assertEqual([json.loads(x) for x in first_complete_lines(path, 2)],
                             [{"i": 0}, {"i": 1}])
            self.assertEqual([json.loads(x) for x in last_complete_lines(path, 2)],
                             [{"i": 1}, {"i": 2}])

    def test_record_retokenization_and_provenance_pass(self):
        errors, summary = record_errors(
            {"content_split": "fixture"}, provenance(1), 1, FakeTokenizer()
        )
        self.assertEqual(errors, [])
        self.assertEqual(set(summary), {
            "stage_record_index", "source", "data_line_sha256",
            "provenance_sha256", "training_token_sha256", "content_tokens",
        })
        self.assertEqual(summary["content_tokens"], SEQ_LEN - 1)
        self.assertEqual(summary["source"], "longcrawl")
        self.assertNotIn("fixture", json.dumps(summary))

    def test_plaintext_and_index_drift_fail(self):
        value = provenance(2)
        value["stage_record_index"] = 3
        value["text"] = "forbidden"
        errors = provenance_errors(value, 2)
        self.assertIn("stage_record_index differs", errors)
        self.assertIn("plaintext key entered provenance", errors)

    def test_frozen_plan_and_cpu_only_guards_are_present(self):
        root = Path(__file__).resolve().parents[1]
        plan = (root /
            "IN_PLACE_TTT_BASELINE_QWEN3_4B_BASE_PARTIAL_CORPUS_SMOKE_ADDENDUM.md"
        ).read_text()
        source = (root / "scripts/audit_in_place_ttt_partial_corpus_smoke.py").read_text()
        self.assertIn("first eight FineWeb/LongCrawl record pairs", plan)
        self.assertIn("latest 32 complete provenance rows", plan)
        self.assertIn("hku-gpu2", plan)
        self.assertIn('os.environ.get("CUDA_VISIBLE_DEVICES")', source)
        self.assertIn('/proc/{args.builder_pid}/cmdline', source)


if __name__ == "__main__":
    unittest.main()
