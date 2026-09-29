import unittest

from tasks.build_cbf_writer_episodes import build
from tests.test_cbf_title_recall import CharacterTokenizer


class WriterEpisodeTests(unittest.TestCase):
    def test_source_split_and_label_exclusion_from_question(self):
        documents = [{"source_id": f"p{i}", "content_split": f"Unique Title {i}\n" + "z" * 80}
                     for i in range(6)]
        metadata = [{"source_id": f"p{i}", "title": f"Unique Title {i}"} for i in range(6)]
        tokenizer = CharacterTokenizer()
        rows = build(documents, metadata, tokenizer, {"train": 2, "dev": 2, "test": 2}, 50)
        self.assertEqual(len({row["group_id"] for row in rows}), 6)
        self.assertEqual(len({tuple(row["query_ids"]) for row in rows}), 1)
        for row in rows:
            self.assertEqual(len(row["context_ids"]), 50)
            self.assertNotIn(tokenizer.decode(row["answer_ids"]).strip(), tokenizer.decode(row["query_ids"]))
        documents[-1]["source_id"] = documents[0]["source_id"]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            build(documents, metadata, tokenizer, {"train": 2, "dev": 2, "test": 2}, 50)


if __name__ == "__main__":
    unittest.main()
