import tempfile
import unittest
import hashlib
from pathlib import Path

from scripts.audit_in_place_ttt_public_corpus import (
    PROFILES as AUDIT_PROFILES,
    SourceContinuity,
    canonical,
)
from scripts.build_in_place_ttt_public_corpus import (
    PROFILES,
    PersistentSourcePacker,
    forbidden_key,
    prepare_root,
    token_hash,
)


class FakeTokenizer:
    eos_token = "<eos>"
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        output = []
        cursor = 0
        while cursor < len(text):
            if text.startswith(self.eos_token, cursor):
                output.append(0); cursor += len(self.eos_token)
            else:
                output.append(ord(text[cursor]) + 1); cursor += 1
        return output

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        del skip_special_tokens, clean_up_tokenization_spaces
        return "".join(self.eos_token if value == 0 else chr(value - 1) for value in ids)


def documents(source, count=8):
    for index in range(count):
        yield {
            "source": source,
            "text": chr(65 + index) * 23,
            "locator": {"file_index": 0, "repository_path": "sample/file.parquet",
                        "parquet_row_index": index},
        }


class PublicCorpusTest(unittest.TestCase):
    def test_prefetched_snapshot_cannot_skip_a_document(self):
        inventory = {"files": [{"repository_path": "sample/file.parquet"}]}
        continuity = SourceContinuity("fineweb", inventory)
        segments = []
        hasher = hashlib.sha256()
        for index in range(2):
            entry = {
                "source": "fineweb",
                "document_index": index,
                "locator": {
                    "file_index": 0,
                    "repository_path": "sample/file.parquet",
                    "parquet_row_index": index,
                },
                "document_sha256": f"{index:064x}",
                "content_token_count": 7,
                "token_count_with_boundary": 8,
            }
            hasher.update((canonical(entry) + "\n").encode())
            segments.append({**entry, "cursor": 0})
        snapshot = {
            "source": "fineweb",
            "documents_loaded": 2,
            "pending_token_count": 16,
            "pending_tokens_sha256": "0" * 64,
            "pending_segments": segments,
            "document_stream_sha256": hasher.hexdigest(),
        }
        self.assertTrue(continuity.validate_snapshot(snapshot))
        skipped = {**segments[1], "token_start": 0, "token_end": 8}
        self.assertEqual(continuity.consume_span(skipped), (False, 0))
        first = {**segments[0], "token_start": 0, "token_end": 8}
        second = {**segments[1], "token_start": 0, "token_end": 8}
        self.assertEqual(continuity.consume_span(first), (True, 8))
        self.assertEqual(continuity.consume_span(second), (True, 8))

    def test_continuity_memory_is_bounded_by_active_documents(self):
        inventory = {"files": [{"repository_path": "sample/file.parquet"}]}
        continuity = SourceContinuity("fineweb", inventory)
        count = 10_000
        for index in range(count):
            span = {
                "source": "fineweb",
                "document_index": index,
                "locator": {
                    "file_index": 0,
                    "repository_path": "sample/file.parquet",
                    "parquet_row_index": index,
                },
                "document_sha256": f"{index:064x}",
                "content_token_count": 7,
                "token_count_with_boundary": 8,
                "token_start": 0,
                "token_end": 8,
            }
            self.assertEqual(continuity.consume_span(span), (True, 8))
        snapshot = {
            "source": "fineweb",
            "documents_loaded": count,
            "pending_token_count": 0,
            "pending_tokens_sha256": "0" * 64,
            "pending_segments": [],
            "document_stream_sha256": continuity.hasher.copy().hexdigest(),
        }
        self.assertTrue(continuity.validate_snapshot(snapshot))
        self.assertEqual(continuity.documents_seen, count)
        self.assertLessEqual(len(continuity.entries), 1)
        self.assertLessEqual(continuity.peak_active_documents, 2)

    def test_packer_preserves_cursor_across_sequence_change(self):
        tokenizer = FakeTokenizer()
        packer = PersistentSourcePacker("fineweb", documents("fineweb"), tokenizer)
        text1, ids1, provenance1 = packer.next_record(16)
        state1 = packer.snapshot()
        text2, ids2, provenance2 = packer.next_record(32)
        state2 = packer.snapshot()
        self.assertEqual(len(tokenizer.encode(text1, add_special_tokens=False)) + 1, 16)
        self.assertEqual(len(tokenizer.encode(text2, add_special_tokens=False)) + 1, 32)
        self.assertEqual(provenance1["training_token_sha256"], token_hash(ids1))
        self.assertEqual(provenance2["training_token_sha256"], token_hash(ids2))
        self.assertGreaterEqual(state2["documents_loaded"], state1["documents_loaded"])
        self.assertNotEqual(state1["pending_tokens_sha256"], token_hash([]))
        self.assertFalse(forbidden_key(provenance1))

        inventory = {"files": [{"repository_path": "sample/file.parquet"}]}
        continuity = SourceContinuity("fineweb", inventory)
        for span in provenance1["spans"]:
            self.assertTrue(continuity.consume_span(span)[0])
        self.assertTrue(continuity.validate_snapshot(state1))
        for span in provenance2["spans"]:
            self.assertTrue(continuity.consume_span(span)[0])
        self.assertTrue(continuity.validate_snapshot(state2))
        packer.close()

    def test_frozen_profiles_match_registered_counts(self):
        self.assertEqual(PROFILES["pilot"][0]["records_per_source"], 64)
        self.assertEqual(PROFILES["pilot"][1]["records_per_source"], 8)
        preliminary = PROFILES["preliminary_100m"]
        self.assertEqual(preliminary[0]["records_per_source"], 864)
        self.assertEqual(preliminary[1]["records_per_source"], 160)
        self.assertEqual(
            [2 * stage["records_per_source"] // 64 for stage in preliminary],
            [27, 5],
        )
        self.assertEqual(
            sum(2 * stage["records_per_source"] * stage["seq_len"]
                for stage in preliminary),
            98_566_144,
        )
        self.assertEqual(
            AUDIT_PROFILES["preliminary_100m"],
            tuple((stage["name"], stage["seq_len"], stage["records_per_source"])
                  for stage in preliminary),
        )
        self.assertEqual(PROFILES["full"][0]["records_per_source"], 305184)
        self.assertEqual(PROFILES["full"][1]["records_per_source"], 57216)
        for profile in ("preliminary_100m", "full"):
            for stage in PROFILES[profile]:
                self.assertEqual(2 * stage["records_per_source"] % 64, 0)

    def test_output_root_is_atomic_and_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "corpus"
            staging = prepare_root(output)
            self.assertEqual(staging.name, "corpus.partial")
            with self.assertRaises(FileExistsError):
                prepare_root(output)
            staging.rmdir(); output.mkdir()
            with self.assertRaises(FileExistsError):
                prepare_root(output)


if __name__ == "__main__":
    unittest.main()
