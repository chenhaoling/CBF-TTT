"""Pure-data checks for the opt-in title-recall protocol and preregistered gate."""

import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_cbf_title_recall_gate import evaluate
from tasks.build_cbf_title_recall_scenarios import build_scenarios


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, ids):
        return "".join(chr(value) for value in ids)


class TitleRecallTests(unittest.TestCase):
    def test_builder_keeps_query_fixed_across_read_conditions(self):
        documents = []
        source_meta = []
        for index in range(12):
            title = f"Unique paper title {index}"
            documents.append({"source_id": str(index), "content_split": title + "\n" + "z" * 80})
            source_meta.append({"source_id": str(index), "title": title})
        rows, meta = build_scenarios(documents, source_meta, CharacterTokenizer(),
                                     {"train": 1, "dev": 1, "test": 1}, context_tokens=50)
        self.assertEqual(len(rows), 12)
        self.assertEqual(meta["read_conditions"], ["kv_intact", "memory_only"])
        for row in rows:
            self.assertEqual(len(row["context_ids"]), 100)
            self.assertEqual([future["reset_kv"] for future in row["futures"]], [False, True])
            self.assertEqual(row["futures"][0]["queries"], row["futures"][1]["queries"])
            self.assertEqual(row["objective"], "joint_title_recall_v1")

    def test_gate_uses_memory_only_and_complete_groups(self):
        rows = []
        for group in range(3):
            for regime in ("both_relevant", "old_only", "new_only", "neither_relevant"):
                memory = {"00": 1.0, "01": 1.0, "10": 1.0, "11": 1.0}
                if regime == "new_only":
                    memory["11"] = 0.98
                elif regime == "old_only":
                    memory.update({"00": 1.02, "11": 1.02})
                elif regime == "neither_relevant":
                    memory.update({"00": 0.98, "11": 1.02})
                rows.append({"id": f"{group}-{regime}", "group_id": f"g{group}",
                             "split": ("train", "dev", "test")[group], "regime": regime,
                             "boundary": 2, "protocol": "joint_title_recall_v1",
                             "future_meta": [{"reset_kv": False}, {"reset_kv": True}],
                             "actions": [[0, 0], [0, 1], [1, 0], [1, 1]],
                             "losses_by_future": [[1.0, memory[key]]
                                                  for key in ("00", "01", "10", "11")]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            result = evaluate([path], expected_groups=3, min_groups=3)
            self.assertTrue(result["passed_joint_gate"])
            self.assertEqual(result["useful_write_groups"], 3)
            self.assertEqual(result["useful_keep_groups"], 3)
            self.assertEqual(result["useful_clear_groups"], 3)
            path.write_text("".join(json.dumps(row) + "\n" for row in rows[:-1]), encoding="utf-8")
            with self.assertRaises(ValueError):
                evaluate([path], expected_groups=3, min_groups=3)


if __name__ == "__main__":
    unittest.main()
