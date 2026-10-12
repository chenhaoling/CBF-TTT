"""Construction, schedule, objective, and prefixed-writer tests."""

import unittest

from tasks.train_final_writer_binding import expand_scenes, schedule


def fake_source_rows():
    rows = []
    choices = list(range(100, 108))
    for group in range(24):
        for variant, label in enumerate((group % 8, (group + 4) % 8)):
            chunks = [[1] * 4096, [2] * 4096]
            chunks[1][-3] = choices[label]
            query = {"ids": [7, 8], "label": label, "answer_id": choices[label], "key": f"k{group}"}
            rows.append({"id": f"old{group}_{variant}", "twin_id": f"old{group}_{1-variant}",
                         "group": group, "variant": variant,
                         "split": "dev" if group < 8 else "q" if group < 16 else "v",
                         "domain": "fineweb" if group % 2 == 0 else "longcrawl",
                         "sources": [f"source{group}"], "chunks": chunks,
                         "choice_ids": choices, "queries": {"target": {"options": query}},
                         "context_sha256": "unused"})
    return rows


class ConstructionTests(unittest.TestCase):
    def test_counterfactual_expansion_changes_only_value(self):
        rows = expand_scenes(fake_source_rows())
        self.assertEqual(len(rows), 192)
        part = [row for row in rows if row["group"] == 3]
        self.assertEqual({row["label"] for row in part}, set(range(8)))
        self.assertEqual(len({tuple(row["query_ids"]) for row in part}), 1)
        for row in part:
            self.assertEqual(row["chunks"][1][-3], row["answer_id"])
            self.assertEqual(row["token_location"], [1, 4093])

    def test_balanced_opposite_schedule(self):
        batches = schedule((0, 1, 2), 8)
        self.assertEqual(len(batches), 96)
        for group in (0, 1, 2):
            labels = [label for g, left, right in batches if g == group for label in (left, right)]
            self.assertEqual({label: labels.count(label) for label in range(8)}, {label: 8 for label in range(8)})
            self.assertTrue(all((right - left) % 8 == 4 for g, left, right in batches if g == group))


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch tests run on the experiment server")
class ObjectiveTests(unittest.TestCase):
    def test_content_separation_reduces_objective(self):
        from tasks.train_final_writer_binding import paired_objective
        choices = list(range(8))
        zero = torch.zeros(16)
        flat_left = torch.zeros(16, requires_grad=True)
        flat_right = torch.zeros(16, requires_grad=True)
        flat, _ = paired_objective(flat_left, flat_right, zero, 0, 4, choices)
        good_left = torch.zeros(16); good_left[0] = 4
        good_right = torch.zeros(16); good_right[4] = 4
        good, metrics = paired_objective(good_left, good_right, zero, 0, 4, choices)
        self.assertLess(float(good), float(flat))
        self.assertEqual(float(metrics["accuracy"]), 1.)

    def test_prefixed_writer_has_live_gradients(self):
        from inference_model.hf_qwen3.configuration_qwen3 import Qwen3Config
        from inference_model.hf_qwen3.modeling_qwen3 import Qwen3ForCausalLM
        from cbf_ttt.event_writer import write_memory_after_prefix
        from scripts.probe_ttt_writer_training import writer_parameters
        config = Qwen3Config(vocab_size=32, hidden_size=8, intermediate_size=16,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
            head_dim=4, max_position_embeddings=64, ttt_mode=True,
            ttt_layers=[0, 1], ttt_chunk=4, ttt_lr=.3, ttt_proj=True)
        config._attn_implementation = "eager"
        model = Qwen3ForCausalLM(config).eval()
        params = writer_parameters(model)
        with torch.no_grad():
            for name, value in params.items():
                if ".ttt_conv." in name: value.normal_(0, .02)
        memory = write_memory_after_prefix(model, [1, 2, 3, 4], [5, 6, 7, 8])
        sum(value.sum() for value in memory.values()).backward()
        self.assertTrue(all(value.grad is not None and torch.isfinite(value.grad).all()
                            for value in params.values()))


if __name__ == "__main__":
    unittest.main()
