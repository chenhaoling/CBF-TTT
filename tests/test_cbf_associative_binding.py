import unittest

from tasks.train_cbf_associative_binding import objective, permuted_contexts, ranking_gate, schedule


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert text.startswith(" code_")
        return [900, int(text[-5:-3]), int(text[-3:])]


def fake_rows(groups=2):
    rows = []
    for group in range(groups):
        for twin in range(2):
            answers = [f"code_{group}{twin}{index:03d}" for index in range(4)]
            suffix = []
            for answer in answers:
                suffix += [11, 12] + FakeTokenizer().encode(" " + answer) + [13]
            context = [1] * 20 + suffix
            for index, answer in enumerate(answers):
                rows.append({
                    "id": f"g{group}.{twin}.q{index}", "group_id": f"g{group}",
                    "context_id": f"g{group}.{twin}", "split": "train",
                    "context_ids": context, "fact_suffix_tokens": len(suffix),
                    "answer": answer, "anchor_query": index == 0,
                })
    return rows


class AssociativeDataTests(unittest.TestCase):
    def test_schedule_keeps_four_question_contexts(self):
        rows = fake_rows()
        batches = schedule(rows, 3, 7)
        self.assertEqual(len(batches), 12)
        self.assertTrue(all(len(batch) == 4 for batch in batches))
        self.assertEqual(len({sample.split(".")[:2][0] + "." + sample.split(".")[1]
                              for sample in batches[0]}), 1)

    def test_permutation_is_length_preserving_derangement(self):
        rows = fake_rows(1)
        contexts, assigned = permuted_contexts(rows, FakeTokenizer())
        self.assertEqual(set(contexts), {"g0.0", "g0.1"})
        for row in rows:
            self.assertEqual(len(contexts[row["context_id"]]), len(row["context_ids"]))
            self.assertNotEqual(assigned[row["id"]], row["answer"])
        self.assertNotEqual(contexts["g0.0"], rows[0]["context_ids"])


class AssociativeObjectiveTests(unittest.TestCase):
    @staticmethod
    def values(torch, digits):
        return [{"digits": torch.tensor(value, requires_grad=True),
                 "prefix": torch.tensor(1.), "eos": torch.tensor(1.),
                 "anchor": index == 0} for index, value in enumerate(digits)]

    def test_all_fact_objective_contrasts_every_question(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        positives = self.values(torch, [1., 2., 3., 4.])
        negatives = [{"digits": torch.tensor(value, requires_grad=True)}
                     for value in [2., 3., 4., 5.]]
        loss, _, contrast, terms = objective(positives, negatives, "all_fact_permutation")
        self.assertEqual(len(terms), 4)
        self.assertGreater(float(contrast), 0)
        loss.backward()
        self.assertTrue(all(value["digits"].grad.item() > 0 for value in positives))
        self.assertTrue(all(value["digits"].grad.item() < 0 for value in negatives))


class AssociativeGateTests(unittest.TestCase):
    def test_gate_requires_both_negative_controls(self):
        dev = {
            "generation": {"full_kv": {"code_correct": .9}},
            "global16": {
                "policies": {"correct": {"top1": .3}, "permuted": {"top1": .1},
                             "wrong": {"top1": .15}},
                "correct_comparisons": {
                    "permuted": {"digit_gain_bootstrap_95ci": [.01, .2]},
                    "wrong": {"digit_gain_bootstrap_95ci": [.02, .3]},
                },
            },
        }
        self.assertTrue(ranking_gate(dev)["passed"])
        dev["global16"]["correct_comparisons"]["wrong"]["digit_gain_bootstrap_95ci"][0] = -.01
        self.assertFalse(ranking_gate(dev)["passed"])


if __name__ == "__main__":
    unittest.main()

