import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import audit_in_place_ttt_2x5090_resource_inputs as audit_module
from scripts.audit_in_place_ttt_2x5090_resource_inputs import (
    SPECS as AUDIT_SPECS,
    SOURCE_PROFILES as AUDIT_SOURCE_PROFILES,
    derive_expected,
    forbidden,
)
from scripts.build_in_place_ttt_2x5090_resource_inputs import (
    SPECS as BUILD_SPECS,
    SOURCE_PROFILES as BUILD_SOURCE_PROFILES,
    has_plaintext,
    stable_prefix,
)


class CharacterTokenizer:
    eos_token = "|"
    eos_token_id = ord("|")

    @staticmethod
    def encode(text, add_special_tokens=False):
        del add_special_tokens
        return [ord(value) for value in text]

    @staticmethod
    def decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        del skip_special_tokens, clean_up_tokenization_spaces
        return "".join(chr(value) for value in ids)


class TwoRTX5090ResourceInputsTest(unittest.TestCase):
    def test_registered_builder_and_auditor_specs_match(self):
        self.assertEqual(BUILD_SPECS, AUDIT_SPECS)
        self.assertEqual(BUILD_SOURCE_PROFILES, AUDIT_SOURCE_PROFILES)
        self.assertIn("preliminary_100m", BUILD_SOURCE_PROFILES)
        self.assertEqual(tuple(BUILD_SPECS), (8192, 16384, 32768, 65536, 131072))

    def test_independent_derivation_matches_builder_for_prefix(self):
        tokenizer = CharacterTokenizer()
        built = stable_prefix("abcdefgh", tokenizer, 4)
        audited = derive_expected("abcdefgh", tokenizer, 4)
        self.assertEqual(built, audited)
        self.assertEqual(built["text"], "abcd")
        self.assertEqual(len(built["training_ids"]), 5)
        self.assertEqual(built["synthetic_eos_tokens"], 0)

    def test_full_parent_is_preserved_byte_for_byte(self):
        tokenizer = CharacterTokenizer()
        text = "a b\nç"
        result = stable_prefix(text, tokenizer, len(text))
        self.assertEqual(result["text"], text)
        self.assertEqual(result["raw_parent_tokens_consumed"], len(text))

    def test_plaintext_metadata_is_rejected_recursively(self):
        value = {"safe": [{"content_split": "leak"}]}
        self.assertTrue(has_plaintext(value))
        self.assertTrue(forbidden(value))
        self.assertFalse(has_plaintext({"content_token_sha256": "abc"}))

    def test_auditor_cli_maps_named_paths_without_type_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "missing_artifacts"
            root.mkdir()
            output = Path(temporary) / "audit.json"
            argv = [
                "audit", "--root", str(root), "--source-root", str(root),
                "--source-audit", str(root / "source_audit.json"),
                "--tokenizer", str(root), "--plan", str(root / "plan.md"),
                "--builder", str(root / "builder.py"), "--output", str(output),
                "--expected-source-manifest-sha256", "a" * 64,
                "--required-source-profile", "full",
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch("builtins.print"), \
                    self.assertRaises(SystemExit) as raised:
                audit_module.main()
            self.assertEqual(raised.exception.code, 1)
            result = json.loads(output.read_text())
            self.assertFalse(result["passed"])
            self.assertIn("missing artifacts", result["errors"][0])


if __name__ == "__main__":
    unittest.main()
