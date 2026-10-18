import argparse
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts import watch_and_run_in_place_ttt_public_corpus_streaming_audit as watcher


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class StreamingFallbackDecisionTest(unittest.TestCase):
    def test_process_classifier_accepts_direct_python_builder(self):
        self.assertTrue(watcher.is_active_corpus_command([
            "/env/bin/python", "scripts/build_in_place_ttt_public_corpus.py",
        ]))

    def test_process_classifier_accepts_time_wrapped_auditor(self):
        self.assertTrue(watcher.is_active_corpus_command([
            "/usr/bin/time", "-v", "/env/bin/python",
            "/repo/scripts/audit_in_place_ttt_public_corpus.py",
        ]))

    def test_process_classifier_ignores_watcher_dependency_paths(self):
        self.assertFalse(watcher.is_active_corpus_command([
            "/env/bin/python", "watch_and_run_streaming_audit.py",
            "--builder", "/old/scripts/build_in_place_ttt_public_corpus.py",
            "--streaming-auditor", "/new/scripts/audit_in_place_ttt_public_corpus.py",
        ]))

    def fixture(self, root: Path):
        corpus = root / "corpus"
        tokenizer = root / "tokenizer"
        corpus.mkdir()
        tokenizer.mkdir()
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            (tokenizer / name).write_text(name)
        plan = root / "plan.md"
        plan.write_text("plan")
        builder = root / "builder.py"
        builder.write_text("builder")
        inventory = root / "inventory.json"
        inventory.write_text("{}")
        inventory_audit = root / "inventory_audit.json"
        inventory_audit.write_text("{}")
        status = root / "status.txt"
        status.write_text("FAILED\n")
        artifacts = (
            "stage1_32k.jsonl",
            "stage1_32k.provenance.jsonl",
            "stage2_128k.jsonl",
            "stage2_128k.provenance.jsonl",
        )
        for name in artifacts:
            (corpus / name).write_text("{}\n")
        manifest = {
            "protocol": watcher.CORPUS_PROTOCOL,
            "profile": "full",
            "totals": dict(watcher.EXPECTED_TOTALS),
            "bindings": {
                "plan": {"sha256": sha(plan)},
                "builder": {"sha256": sha(builder)},
                "inventory": {"sha256": sha(inventory)},
                "inventory_audit": {"sha256": sha(inventory_audit)},
                "tokenizer_path": str(tokenizer.resolve()),
                "tokenizer_assets": watcher.tokenizer_assets(tokenizer),
            },
        }
        (corpus / "manifest.json").write_text(json.dumps(manifest))
        return argparse.Namespace(
            corpus_root=corpus,
            tokenizer=tokenizer,
            plan=plan,
            builder=builder,
            inventory=inventory,
            inventory_audit=inventory_audit,
            original_status=status,
        )

    def test_failed_original_after_publication_releases_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(Path(temporary))
            self.assertEqual(watcher.decide(args, processes=[]).action, "run_fallback")

    def test_active_original_auditor_keeps_waiting(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(Path(temporary))
            decision = watcher.decide(args, processes=[{"pid": 9, "argv": ["audit"]}])
            self.assertEqual(decision.action, "wait")

    def test_valid_existing_audit_is_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(Path(temporary))
            manifest = args.corpus_root / "manifest.json"
            audit = {
                "protocol": watcher.AUDIT_PROTOCOL,
                "profile": "full",
                "passed": True,
                "checks": {"all": True},
                "bindings": {
                    "manifest_sha256": sha(manifest),
                    "auditor_sha256": watcher.EXPECTED_HASHES["original_auditor"],
                },
            }
            (args.corpus_root / "independent_audit.json").write_text(json.dumps(audit))
            self.assertEqual(watcher.decide(args, processes=[]).action, "accept_existing")

    def test_existing_failed_audit_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(Path(temporary))
            (args.corpus_root / "independent_audit.json").write_text(json.dumps({
                "protocol": watcher.AUDIT_PROTOCOL,
                "profile": "full",
                "passed": False,
                "checks": {"all": False},
                "bindings": {},
            }))
            self.assertEqual(watcher.decide(args, processes=[]).action, "refuse")

    def test_failed_wrapper_without_published_manifest_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(Path(temporary))
            (args.corpus_root / "manifest.json").unlink()
            self.assertEqual(watcher.decide(args, processes=[]).action, "refuse")

    def test_completed_wrapper_without_audit_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(Path(temporary))
            args.original_status.write_text("COMPLETED\n")
            self.assertEqual(watcher.decide(args, processes=[]).action, "refuse")


if __name__ == "__main__":
    unittest.main()
