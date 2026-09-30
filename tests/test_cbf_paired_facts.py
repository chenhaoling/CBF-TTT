import copy
import re
import unittest

from tasks.build_cbf_paired_facts import build, validate


class WordTokenizer:
    def __init__(self):
        self.vocab = {}

    def encode(self, text, add_special_tokens=False):
        return [self.vocab.setdefault(word, len(self.vocab)) for word in re.findall(r" ?\w+|[^\w]", text)]


class PairedFactTests(unittest.TestCase):
    def test_counterfactual_balance_and_source_isolation(self):
        documents = [{"source_id": f"p{i}", "content_split": " background"*150} for i in range(24)]
        rows = build(documents, WordTokenizer(), {"train": 8, "dev": 8, "test": 8}, chunk_size=64)
        self.assertEqual(len(rows), 48)
        self.assertTrue(all(len(r["context_ids"]) == 64 for r in rows))
        by_id = {r["id"]: r for r in rows}
        for row in rows:
            twin = by_id[row["twin_id"]]
            self.assertEqual(sum(a != b for a, b in zip(row["context_ids"], twin["context_ids"])), 1)
        broken = copy.deepcopy(rows)
        broken[0]["split"] = "test"
        with self.assertRaisesRegex(ValueError, "cross-split"):
            validate(broken)
        broken = copy.deepcopy(rows)
        broken[0]["answer_ids"] = [999]
        with self.assertRaisesRegex(ValueError, "answer index"):
            validate(broken)


if __name__ == "__main__":
    unittest.main()
