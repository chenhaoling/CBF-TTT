"""Check bounded writer gradients and frozen-backbone training end to end."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from cbf_ttt.writer import LowRankWriter, memory_nll
from tasks.cbf_writer import extract, train, evaluate_test


class WriterTests(unittest.TestCase):
    def test_identity_cap_and_input_detachment(self):
        torch.manual_seed(123)
        writer = LowRankWriter({0: (4, 6)}, rank=2)
        raw = torch.randn(4, 6, requires_grad=True)
        self.assertTrue(torch.equal(writer({0: raw})[0], raw))
        writer({0: raw})[0].square().sum().backward()
        self.assertIsNone(raw.grad)
        with torch.no_grad():
            writer.b["0"].normal_(0, 2)
        self.assertLessEqual(float(writer({0: raw})[0].detach().norm()), float(raw.detach().norm()) + 1e-5)

    def test_tiny_training_changes_writer_but_freezes_backbone(self):
        from inference_model.hf_qwen3.configuration_qwen3 import Qwen3Config
        from inference_model.hf_qwen3.modeling_qwen3 import Qwen3ForCausalLM

        torch.manual_seed(121)
        config = Qwen3Config(vocab_size=32, hidden_size=8, intermediate_size=16,
                             num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                             head_dim=4, max_position_embeddings=32, ttt_mode=True,
                             ttt_layers=[0, 1], ttt_chunk=4, ttt_lr=0.1, ttt_proj=True)
        model = Qwen3ForCausalLM(config).eval().requires_grad_(False)
        with torch.no_grad():
            for layer in config.ttt_layers:
                model.model.layers[layer].mlp.ttt_conv.weight.normal_(0, 0.02)
        original = {key: value.clone() for key, value in model.state_dict().items()}
        episodes = [{"id": f"p{i}", "group_id": f"p{i}", "protocol": "task_writer_v1",
                     "split": split, "context_ids": [1+i, 2+i, 3+i, 4+i],
                     "query_ids": [9], "answer_ids": [10+i, 11+i]}
                    for i, split in enumerate(("train", "train", "dev", "test", "test"))]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config.to_json_file(root / "config.json")
            data = root / "episodes.jsonl"
            data.write_text("".join(json.dumps(row) + "\n" for row in episodes))
            args = SimpleNamespace(model=str(root), data=str(data), features=str(root / "features"),
                                   shard=0, shards=1, output=str(root / "training"),
                                   seed=123, rank=2, lr=0.01, epochs=2, smoke=True)
            extract(args, model, episodes)
            train(args, model, episodes)
            selection = json.loads((root / "training/selection.json").read_text())
            self.assertEqual(selection["optimizer_steps"], 1)
            self.assertFalse(selection["passed_dev_gate"])
            self.assertTrue(all(torch.equal(original[key], value) for key, value in model.state_dict().items()))
            for parameter in model.parameters():
                self.assertIsNone(parameter.grad)
            args.checkpoint_dir = str(root / "training")
            args.output = str(root / "test")
            with self.assertRaisesRegex(ValueError, "dev stage gate failed"):
                evaluate_test(args, model, episodes)
            self.assertFalse((root / "test").exists())
            # Synthetic fixture exercises the successful-gate evaluation branch without tuning real test data.
            selection.update({"passed_dev_gate": True, "smoke": False})
            (root / "training/selection.json").write_text(json.dumps(selection))
            evaluate_test(args, model, episodes)
            result = json.loads((root / "test/summary.json").read_text())
            self.assertEqual(result["test_papers"], 2)
            self.assertEqual(set(result["writer_gain_vs_controls"]), {"none", "raw", "mismatched", "train_mean"})


if __name__ == "__main__":
    unittest.main()
