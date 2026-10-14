import unittest

from tasks.diagnose_qwen_sparse_reader import (
    READERS,
    apply_retention,
    select_reader,
    sparse_output,
)


class FakeEmbeddings:
    def __init__(self, values):
        self.seed = 1
        self.dim = 3
        self.values = values

    def get(self, kind, text):
        return self.values[kind, text]


class QwenSparseReaderTest(unittest.TestCase):
    def setUp(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed locally")
        self.torch = torch
        values = {
            ("entity", "a"): torch.tensor([1.0, 0.0, 0.0]),
            ("entity", "b"): torch.tensor([0.0, 1.0, 0.0]),
            ("entity", "q"): torch.tensor([0.8, 0.6, 0.0]),
            ("value", "x"): torch.tensor([1.0, 0.0, 0.0]),
            ("value", "y"): torch.tensor([0.0, 1.0, 0.0]),
        }
        self.embeddings = FakeEmbeddings(values)

    def state(self):
        torch = self.torch
        return [
            {
                "batch": "old",
                "facts": {"a": "x"},
                "delta": torch.outer(
                    self.embeddings.get("value", "x"),
                    self.embeddings.get("entity", "a"),
                ),
                "weight": 1.0,
                "block_index": 0,
            },
            {
                "batch": "new",
                "facts": {"b": "y"},
                "delta": 1.15
                * torch.outer(
                    self.embeddings.get("value", "y"),
                    self.embeddings.get("entity", "b"),
                ),
                "weight": 1.0,
                "block_index": 1,
            },
        ]

    def test_registered_reader_order(self):
        self.assertEqual(READERS[0], "dense_all")
        self.assertEqual(READERS[-1], "exact_key_upper")

    def test_topk_uses_highest_key_cosine(self):
        output, used = sparse_output(
            self.state(), {"key": "q"}, self.embeddings, "component_top1"
        )
        self.assertEqual(used, 1)
        self.assertTrue(self.torch.allclose(output, self.torch.tensor([0.8, 0.0, 0.0])))

    def test_zero_weight_components_are_inactive(self):
        state = self.state()
        state[0]["weight"] = 0.0
        output, used = sparse_output(
            state, {"key": "q"}, self.embeddings, "component_top1"
        )
        self.assertEqual(used, 1)
        self.assertTrue(
            self.torch.allclose(output, self.torch.tensor([0.0, 0.69, 0.0]))
        )

    def test_retention_tracks_history_last2_last1(self):
        state = self.state()
        extra = dict(state[-1])
        extra["batch"] = "latest"
        state.append(extra)
        assignment = {
            "batch": "append",
            "facts": {"a": "x"},
            "old_refs": [],
            "prelude_lines": 0,
        }
        output = apply_retention(
            state, (0.25, 0.5, 0.75), assignment, state[0]["delta"], 2
        )
        self.assertEqual([row["weight"] for row in output], [0.25, 0.5, 0.75, 1.0])

    def test_reader_selection_prefers_near_best_smaller_k(self):
        seeds = (2301, 2307, 2311)

        def row(accuracy, historical=0.92):
            return {
                "mean_accuracy": accuracy,
                "roles": {"current": 0.99, "historical": historical, "anchor": 0.99},
                "families": {"a": 0.9},
                "per_seed": {str(seed): {"accuracy": accuracy} for seed in seeds},
            }

        summary = {reader: {"oracle_grid": row(0.94)} for reader in READERS}
        summary["dense_all"]["oracle_grid"] = row(0.89, 0.7)
        summary["component_top1"]["oracle_grid"] = row(0.961)
        summary["component_top2"]["oracle_grid"] = row(0.964)
        summary["component_top4"]["oracle_grid"] = row(0.98)
        summary["exact_key_upper"]["oracle_grid"] = row(0.99)
        decision = select_reader(summary)
        self.assertTrue(decision["passed"])
        self.assertEqual(decision["selected"], "component_top4")


if __name__ == "__main__":
    unittest.main()
