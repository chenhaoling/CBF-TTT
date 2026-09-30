import json
import http.client
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.audit_postcutoff_corpus import audit
from scripts.download_postcutoff_arxiv import clean_pdf_text, fetch_pdf
from scripts.evaluate_cbf_postcutoff_gate import evaluate


class PostcutoffPilotTests(unittest.TestCase):
    def test_partial_pdf_transfer_retries_without_caching_partial_bytes(self):
        broken, complete = MagicMock(), MagicMock()
        broken.__enter__.return_value.read.side_effect = http.client.IncompleteRead(b"%PDF-partial", 10)
        complete.__enter__.return_value.read.return_value = b"%PDF-complete"
        with tempfile.TemporaryDirectory() as directory, patch("scripts.download_postcutoff_arxiv.time.sleep"), patch(
            "scripts.download_postcutoff_arxiv.urllib.request.urlopen", side_effect=[broken, complete]
        ) as request:
            path = Path(directory)/"paper.pdf"
            self.assertEqual(fetch_pdf("https://example.org/paper.pdf", path), b"%PDF-complete")
            self.assertEqual(path.read_bytes(), b"%PDF-complete")
            self.assertEqual(request.call_count, 2)

    def test_pdf_cleanup_and_title_scan(self):
        self.assertEqual(clean_pdf_text("hy-\nphen\n\n\nBody\nReferences\nignored"),
                         "hyphen\n\nBody")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            meta = root / "source.json"
            corpus = root / "corpus.jsonl"
            meta.write_text(json.dumps({"accepted": [
                {"source_id": "one", "title": "A New Distinctive Paper Title"},
                {"source_id": "two", "title": "Another Distinctive Paper Title"},
            ]}))
            corpus.write_text('{"content_split":"a new distinctive paper title"}\n')
            result = audit(meta, corpus)
            self.assertEqual(result["matched_source_ids"], ["one"])

    def test_gate_isolates_write_from_retention(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.jsonl"
            rows = []
            for group in range(3):
                for regime in ("both_relevant", "new_only", "old_only", "neither_relevant"):
                    # 10->11 isolates writing; 00->10 isolates retention.
                    corners = {"00": 1.0, "01": 1.0, "10": 1.0, "11": 1.0}
                    if regime == "new_only":
                        corners["11"] = 0.98
                    if regime == "old_only":
                        corners["00"] = 1.02
                        corners["11"] = 1.03
                    if regime == "neither_relevant":
                        corners["10"] = 1.02
                        corners["11"] = 1.05
                    rows.append({"id": f"{group}-{regime}", "group_id": str(group),
                                 "split": "train", "regime": regime,
                                 "protocol": "joint_postcutoff_v1", "corner_losses": corners})
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            result = evaluate([path], expected_groups=3)
            self.assertTrue(result["passed_joint_gate"])
            self.assertEqual(result["useful_write_groups"], 3)
            self.assertEqual(result["harmful_write_groups"], 3)
            rows[0]["split"] = "test"
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            with self.assertRaisesRegex(ValueError, "crosses splits"):
                evaluate([path], expected_groups=3)


if __name__ == "__main__":
    unittest.main()
