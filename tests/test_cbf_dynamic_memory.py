"""Dynamic-world natural packing, matched exposure, and fixed gate checks."""

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from tasks.build_cbf_event_curriculum import materialize
from tasks.pack_cbf_dynamic_memory import pack
from tasks.train_cbf_dynamic_memory import content_gate, schedule, select_view


class CharacterTokenizer:
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [ord(character) + 2 for character in text]

    def decode(self, ids):
        return "".join(chr(token - 2) for token in ids)


class DynamicMemoryTests(unittest.TestCase):
    def build(self, root: Path):
        source = root / "source"
        corpus = root / "mixed.jsonl"
        output = root / "packed"
        materialize(20261012, {"train": 2, "dev": 2, "test": 1}, source)
        # Packing must not open any test example; only its count stays in manifest.
        for path in source.glob("test.*.jsonl"):
            path.unlink()
        records = []
        for index in range(8):
            source_name = "fineweb_edu" if index % 2 == 0 else "longcrawl64"
            text = (f"Natural document {index}. " + "background prose " * 100)
            records.append({"content_split": text, "source": source_name})
        corpus.write_text("".join(json.dumps(row) + "\n" for row in records))
        manifest = pack(source, output, CharacterTokenizer(), str(root / "tokenizer"), corpus, 512)
        rows = [json.loads(line) for line in (output / "rows.jsonl").read_text().splitlines()]
        return source, output, manifest, rows

    def test_natural_pack_keeps_test_sealed_and_twins_causal(self):
        with tempfile.TemporaryDirectory() as temp:
            _, _, manifest, rows = self.build(Path(temp))
            self.assertEqual(manifest["rows"], {"train": 16, "dev": 16})
            self.assertEqual(manifest["background_sources"], {"fineweb_edu": 2, "longcrawl64": 2})
            self.assertFalse(manifest["test_tokenized"])
            self.assertEqual({row["split"] for row in rows}, {"train", "dev"})
            self.assertTrue(all(len(row["context_ids"]) == 512 for row in rows))
            self.assertEqual(len({row["background_sha256"] for row in rows}), 4)
            by_id = {row["id"]: row for row in rows}
            for row in rows:
                twin = by_id[row["twin_row_id"]]
                self.assertEqual(row["query_ids"], twin["query_ids"])
                self.assertEqual(row["answer"] != twin["answer"], row["anchor_query"])
                self.assertNotEqual(row["group_id"], row["wrong_context_id"].split(".")[0])

    def test_question_exposure_is_matched(self):
        with tempfile.TemporaryDirectory() as temp:
            _, _, _, rows = self.build(Path(temp))
            rows, groups = select_view(rows, train_groups=2, dev_groups=2)
            self.assertEqual({key: len(value) for key, value in groups.items()}, {"train": 2, "dev": 2})
            sequential = schedule(rows, "sequential", rounds=3, seed=503)
            joint = schedule(rows, "joint_context", rounds=3, seed=503)
            self.assertEqual(len(sequential), 48)
            self.assertEqual(len(joint), 12)
            self.assertTrue(all(len(batch) == 1 for batch in sequential))
            self.assertTrue(all(len(batch) == 4 for batch in joint))
            for batches in (sequential, joint):
                counts = Counter(sample_id for batch in batches for sample_id in batch)
                self.assertEqual(set(counts.values()), {3})
                self.assertEqual(sum(counts.values()), 48)

    def test_fixed_content_gate(self):
        group = {
            "correct": {"digit_nll": 0.5},
            "wrong": {"digit_nll": 0.8},
            "empty": {"digit_nll": 0.7},
        }
        dev = {
            "means": {
                "correct": {"code_correct": 0.75},
                "wrong": {"code_correct": 0.25},
                "empty": {"code_correct": 0.50},
                "full_kv": {"code_correct": 1.0},
            },
            "groups": {"g0": group, "g1": group},
        }
        self.assertTrue(content_gate(dev)["passed"])
        dev["means"]["correct"]["code_correct"] = 0.55
        self.assertFalse(content_gate(dev)["passed"])

    def test_evaluation_memory_closure_is_four_contexts_per_group(self):
        with tempfile.TemporaryDirectory() as temp:
            _, _, _, rows = self.build(Path(temp))
            rows, _ = select_view(rows, train_groups=2, dev_groups=2)
            for split in ("train", "dev"):
                split_rows = [row for row in rows if row["split"] == split]
                for group in {row["group_id"] for row in split_rows}:
                    group_rows = [row for row in split_rows if row["group_id"] == group]
                    needed = {
                        context_id for row in group_rows
                        for context_id in (row["context_id"], row["wrong_context_id"], row["twin_context_id"])
                    }
                    self.assertEqual(len(needed), 4)


if __name__ == "__main__":
    unittest.main()
