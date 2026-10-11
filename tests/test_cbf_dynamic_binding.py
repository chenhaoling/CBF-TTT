"""Schedule and causal objective tests for dynamic binding training."""

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from tasks.build_cbf_event_curriculum import materialize
from tasks.pack_cbf_dynamic_memory import pack
from tasks.train_cbf_dynamic_binding import objective, schedule
from tasks.train_cbf_dynamic_memory import select_view
from tests.test_cbf_dynamic_memory import CharacterTokenizer


class BindingScheduleTests(unittest.TestCase):
    def test_four_question_joint_schedule(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, output, corpus = root / "source", root / "packed", root / "corpus.jsonl"
            materialize(20261013, {"train": 2, "dev": 2, "test": 1}, source)
            for path in source.glob("test.*.jsonl"):
                path.unlink()
            records = [
                {"content_split": f"document {index} " + "natural words " * 100,
                 "source": "fineweb_edu" if index % 2 == 0 else "longcrawl64"}
                for index in range(8)
            ]
            corpus.write_text("".join(json.dumps(row) + "\n" for row in records))
            pack(source, output, CharacterTokenizer(), str(root / "tokenizer"), corpus, 512)
            rows = [json.loads(line) for line in (output / "rows.jsonl").read_text().splitlines()]
            rows, _ = select_view(rows, 2, 2)
            batches = schedule(rows, rounds=3, seed=607)
            self.assertEqual(len(batches), 12)
            self.assertTrue(all(len(batch) == 4 for batch in batches))
            counts = Counter(sample_id for batch in batches for sample_id in batch)
            self.assertEqual(set(counts.values()), {3})


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch tests run on experiment server")
class BindingObjectiveTests(unittest.TestCase):
    def values(self):
        output = []
        for index in range(4):
            output.append({key: torch.tensor(1., requires_grad=True)
                           for key in ("ce", "digits", "prefix", "eos")})
            output[-1]["anchor"] = index == 0
        return output

    def test_ce_is_mean_question_ce(self):
        values = self.values()
        loss, _, contrast = objective(values, None, "joint_ce")
        self.assertEqual(float(loss), 1.)
        self.assertEqual(float(contrast), 0.)
        loss.backward()
        self.assertTrue(all(value["ce"].grad == .25 for value in values))

    def test_binding_pushes_own_and_twin_anchor_apart(self):
        values = self.values()
        cross = {"digits": torch.tensor(1., requires_grad=True)}
        loss, balanced, contrast = objective(values, cross, "joint_binding")
        self.assertGreater(float(contrast), 0.)
        self.assertEqual(float(balanced), 1.)
        loss.backward()
        self.assertGreater(float(values[0]["digits"].grad), 0.)
        self.assertLess(float(cross["digits"].grad), 0.)
        self.assertTrue(all(value["ce"].grad is None for value in values))


if __name__ == "__main__":
    unittest.main()
