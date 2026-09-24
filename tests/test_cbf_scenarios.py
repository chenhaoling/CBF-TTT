"""Scenario construction checks that do not need model weights or PyTorch."""

import json
from pathlib import Path
import tempfile
import unittest

from tasks.build_cbf_scenarios import _fit_chunk, build_scenarios, load_sources, synthetic_sources


class WordTokenizer:
    bos_token_id = 0

    def __init__(self):
        self.ids = {}

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        result = []
        for word in text.split():
            if word not in self.ids:
                self.ids[word] = len(self.ids) + 1
            result.append(self.ids[word])
        return result


class ScenarioTests(unittest.TestCase):
    def test_group_split_lengths_queries_and_reproducibility(self):
        sources = synthetic_sources(6, 17)
        settings = dict(
            split_counts={"train": 2, "dev": 2, "test": 2}, variants_per_group=3,
            chunk_size=64, context_chunks=2, future_chunks=1,
            futures_per_scenario=2, seed=23,
        )
        scenarios, metadata = build_scenarios(WordTokenizer(), sources, **settings)
        replay, replay_metadata = build_scenarios(WordTokenizer(), sources, **settings)
        self.assertEqual((scenarios, metadata), (replay, replay_metadata))
        self.assertEqual(len(scenarios), 18)
        self.assertEqual({scenario["regime"] for scenario in scenarios},
                         {"stable", "correction", "topic_shift"})
        split_by_group = {}
        for scenario in scenarios:
            split_by_group.setdefault(scenario["group_id"], set()).add(scenario["split"])
            self.assertEqual(len(scenario["context_ids"]), 128)
            self.assertEqual(scenario["context_ids"][0], 0)
            self.assertEqual(len(scenario["futures"]), 2)
            for future in scenario["futures"]:
                self.assertEqual(len(future["continuation_ids"]), 64)
                self.assertEqual({query["kind"] for query in future["queries"]},
                                 {"historical", "current_or_new"})
                self.assertTrue(all(query["query_ids"] and query["answer_ids"] for query in future["queries"]))
        self.assertTrue(all(len(splits) == 1 for splits in split_by_group.values()))

    def test_fact_header_cannot_be_truncated(self):
        with self.assertRaisesRegex(ValueError, "increase --chunk-size"):
            _fit_chunk(WordTokenizer(), "one two three four", "noise", 2)

    def test_structured_source_input(self):
        source = synthetic_sources(1, 3)[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "facts.jsonl"
            path.write_text(json.dumps(source) + "\n", encoding="utf-8")
            self.assertEqual(load_sources(str(path)), [source])
            path.write_text(json.dumps(source) + "\n" + json.dumps(source) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate source group"):
                load_sources(str(path))


if __name__ == "__main__":
    unittest.main()
