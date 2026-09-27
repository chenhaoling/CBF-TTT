"""Scenario construction checks that do not need model weights or PyTorch."""

import json
from pathlib import Path
import tempfile
import unittest

from tasks.build_cbf_scenarios import _fit_chunk, build_scenarios, load_backgrounds, load_sources, synthetic_sources
from scripts.shard_cbf_scenarios import shard_scenarios
from scripts.summarize_cbf_formal_labels import summarize


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

    def test_group_sharding_preserves_all_scenarios_and_split_isolation(self):
        rows = [
            {"id": f"{split}-{group}-v{variant}", "group_id": f"{split}-{group}", "split": split}
            for split in ("train", "dev", "test")
            for group in range(5)
            for variant in range(3)
        ]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "scenarios.jsonl"
            source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            destination = Path(directory) / "shards"
            manifest = shard_scenarios(source, destination, 2)
            self.assertEqual(manifest["scenarios"], 45)
            seen = set()
            for split in ("train", "dev", "test"):
                self.assertEqual(manifest["splits"][split]["groups_per_shard"], [3, 2])
                for shard in range(2):
                    shard_rows = [json.loads(line) for line in
                                  (destination / f"{split}_shard{shard}.jsonl").read_text().splitlines()]
                    self.assertTrue(all(row["split"] == split for row in shard_rows))
                    self.assertTrue(all(int(row["group_id"].rsplit("-", 1)[-1]) % 2 == shard
                                        for row in shard_rows))
                    for row in shard_rows:
                        self.assertNotIn(row["id"], seen)
                        seen.add(row["id"])
            self.assertEqual(seen, {row["id"] for row in rows})

    def test_distinct_natural_background_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "background.jsonl"
            path.write_text("".join(json.dumps({"content_split": f"background{i}"}) + "\n"
                                    for i in range(6)), encoding="utf-8")
            backgrounds = load_backgrounds(str(path), 6)
            self.assertEqual(backgrounds, [f"background{i}" for i in range(6)])
            with self.assertRaisesRegex(ValueError, "needs 7"):
                load_backgrounds(str(path), 7)
            scenarios, metadata = build_scenarios(
                WordTokenizer(), synthetic_sources(3, 4),
                {"train": 1, "dev": 1, "test": 1}, 1, 64, 1, 1, 1, 4,
                backgrounds=[f"background{i}" for i in range(6)],
            )
            self.assertEqual(metadata["background_records"], 6)
            self.assertEqual(len({row["context_ids"][-1] for row in scenarios}), 3)

    def test_first_only_fact_placement_keeps_fact_out_of_later_context_chunks(self):
        tokenizer = WordTokenizer()
        sources = synthetic_sources(3, 5)
        scenarios, metadata = build_scenarios(
            tokenizer, sources, {"train": 1, "dev": 1, "test": 1},
            1, 64, 2, 1, 1, 5, fact_placement="first_only",
        )
        self.assertEqual(metadata["fact_placement"], "first_only")
        source_by_group = {source["group_id"]: source for source in sources}
        for scenario in scenarios:
            fact = source_by_group[scenario["group_id"]]["original_value"]
            fact_tokens = {token_id for word, token_id in tokenizer.ids.items() if fact in word}
            self.assertTrue(fact_tokens.intersection(scenario["context_ids"][:64]))
            self.assertFalse(fact_tokens.intersection(scenario["context_ids"][64:]))

    def test_short_tail_future_preserves_fact_without_full_update(self):
        scenarios, metadata = build_scenarios(
            WordTokenizer(), synthetic_sources(3, 8),
            {"train": 1, "dev": 1, "test": 1},
            1, 64, 2, 1, 1, 8, future_mode="short_tail",
        )
        self.assertEqual(metadata["future_mode"], "short_tail")
        self.assertTrue(all(0 < len(row["futures"][0]["continuation_ids"]) < 64
                            for row in scenarios))

    def test_formal_summary_rejects_incomplete_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard_dir = root / "shards"
            shard_dir.mkdir()
            for split in ("train", "dev", "test"):
                scenario = {"id": split, "group_id": split, "split": split, "regime": "stable"}
                (shard_dir / f"{split}_shard0.jsonl").write_text(json.dumps(scenario) + "\n")
                label = {
                    "id": split, "group_id": split, "split": split, "boundary": 2,
                    "grid": [0.0, 0.5, 1.0], "losses": [1.0, 0.8, 0.9],
                    "alpha_star": 0.5, "benefits": [0.0, 0.2, 0.1],
                    "label_time_s": 1.0, "peak_allocated_gib": 12.0,
                    "peak_reserved_gib": 14.0,
                }
                output = root / f"{split}_shard0_labels.jsonl"
                output.write_text(json.dumps(label) + "\n")
                Path(str(output) + ".summary.json").write_text("{}")
            result = summarize(root, shards=1)
            self.assertEqual(result["scenarios_and_labels"], 3)
            self.assertEqual(result["alpha_counts_by_split"]["train"], {"0.5": 1})
            (root / "test_shard0_labels.jsonl.summary.json").unlink()
            with self.assertRaisesRegex(ValueError, "collector did not finish"):
                summarize(root, shards=1)


if __name__ == "__main__":
    unittest.main()
