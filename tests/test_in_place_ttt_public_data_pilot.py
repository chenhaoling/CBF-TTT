import tempfile
import unittest
from pathlib import Path

from scripts.build_in_place_ttt_public_data_pilot import (
    contains_forbidden_provenance_key,
    iter_packed_records,
    token_ids_sha256,
    validate_output_paths,
)


class FakeTokenizer:
    eos_token = "<eos>"
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        result = []
        cursor = 0
        while cursor < len(text):
            if text.startswith(self.eos_token, cursor):
                result.append(self.eos_token_id)
                cursor += len(self.eos_token)
            else:
                result.append(ord(text[cursor]) + 1)
                cursor += 1
        return result

    def decode(self, token_ids, skip_special_tokens=False,
               clean_up_tokenization_spaces=False):
        del skip_special_tokens, clean_up_tokenization_spaces
        return "".join(self.eos_token if token_id == self.eos_token_id
                       else chr(token_id - 1) for token_id in token_ids)


class PublicDataPilotTest(unittest.TestCase):
    def test_exact_veomni_length_and_contiguous_provenance(self):
        tokenizer = FakeTokenizer()
        documents = [
            {"text": "abcdefghij", "locator": {"stream_row_index": 0}},
            {"text": "klmnopqrstuv", "locator": {"stream_row_index": 1}},
            {"text": "wxyz0123456789", "locator": {"stream_row_index": 2}},
        ]
        records = list(iter_packed_records(documents, tokenizer, seq_len=16,
                                           max_records=2))
        self.assertEqual(len(records), 2)
        cursors = {}
        for record_index, (row, provenance) in enumerate(records):
            content_ids = tokenizer.encode(row["content_split"], add_special_tokens=False)
            training_ids = content_ids + [tokenizer.eos_token_id]
            self.assertEqual(len(content_ids), 15)
            self.assertEqual(len(training_ids), 16)
            self.assertEqual(provenance["record_index"], record_index)
            self.assertEqual(provenance["training_token_sha256"],
                             token_ids_sha256(training_ids))
            self.assertFalse(contains_forbidden_provenance_key(provenance))
            consumed = 0
            for span in provenance["spans"]:
                document_index = span["document_index"]
                self.assertEqual(span["token_start"], cursors.get(document_index, 0))
                cursors[document_index] = span["token_end"]
                consumed += span["token_end"] - span["token_start"]
            self.assertEqual(consumed, provenance["raw_source_tokens_consumed"])

    def test_short_source_fails_instead_of_duplicating_text(self):
        with self.assertRaisesRegex(RuntimeError, "source ended"):
            list(iter_packed_records(
                [{"text": "short", "locator": {"stream_row_index": 0}}],
                FakeTokenizer(), seq_len=16, max_records=1,
            ))

    def test_nested_plaintext_keys_are_rejected(self):
        self.assertTrue(contains_forbidden_provenance_key(
            {"spans": [{"locator": {"text": "leak"}}]}
        ))
        self.assertTrue(contains_forbidden_provenance_key({"content_split": "leak"}))
        self.assertFalse(contains_forbidden_provenance_key(
            {"spans": [{"document_sha256": "abc", "token_start": 0}]}
        ))

    def test_output_and_partial_directories_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "pilot"
            staging = validate_output_paths(output)
            self.assertEqual(staging.name, "pilot.partial")
            with self.assertRaises(FileExistsError):
                validate_output_paths(output)
            staging.rmdir()
            output.mkdir()
            with self.assertRaises(FileExistsError):
                validate_output_paths(output)


if __name__ == "__main__":
    unittest.main()
