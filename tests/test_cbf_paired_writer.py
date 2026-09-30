import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from cbf_ttt.writer import memory_choice_nll, memory_nll
from tasks.cbf_paired_writer import decision, paired_loss, train, test as evaluate_test
from tasks.cbf_writer import extract


class PairedWriterTests(unittest.TestCase):
    def test_pair_gradient_and_content_gate(self):
        correct, wrong = torch.tensor(2.0, requires_grad=True), torch.tensor(2.0, requires_grad=True)
        paired_loss(correct, wrong).backward()
        self.assertEqual(float(correct.grad), 2.0)
        self.assertEqual(float(wrong.grad), -1.0)
        records = [{"group_id": f"p{i//2}", "policies": {
            name: {"nll": 1.0 if name == "writer_1.0" else 2.0, "correct": int(name == "writer_1.0")}
            for name in ("none", "raw_0.5", "writer_1.0", "twin_1.0", "train_mean_1.0")}}
            for i in range(4)]
        self.assertTrue(decision(records, 1.0, 0.5)["passed"])
        for row in records:
            row["policies"]["twin_1.0"] = row["policies"]["writer_1.0"].copy()
        self.assertFalse(decision(records, 1.0, 0.5)["passed"])

    def test_tiny_paired_training_and_test_guard(self):
        from inference_model.hf_qwen3.configuration_qwen3 import Qwen3Config
        from inference_model.hf_qwen3.modeling_qwen3 import Qwen3ForCausalLM
        torch.manual_seed(134)
        config = Qwen3Config(vocab_size=32, hidden_size=8, intermediate_size=16,
                            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                            head_dim=4, max_position_embeddings=32, ttt_mode=True,
                            ttt_layers=[0, 1], ttt_chunk=4, ttt_lr=0.1, ttt_proj=True)
        model = Qwen3ForCausalLM(config).eval().requires_grad_(False)
        with torch.no_grad():
            for layer in config.ttt_layers:
                model.model.layers[layer].mlp.ttt_conv.weight.normal_(0, 0.02)
        original = {k: v.clone() for k, v in model.state_dict().items()}
        choices = list(range(10, 18))
        self.assertAlmostEqual(float(memory_choice_nll(model, {}, [9], choices)[0]),
                               float(memory_nll(model, {}, [9], [10])), places=6)
        rows = []
        for i, split in enumerate(("train", "dev", "test")):
            for j in range(2):
                rows.append({"id": f"p{i}{j}", "group_id": f"p{i}", "twin_id": f"p{i}{1-j}",
                             "split": split, "context_ids": [1, 2, 3, 4+j], "query_ids": [9],
                             "answer_ids": [choices[j]], "choice_ids": choices, "answer_index": j})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config.to_json_file(root/"config.json")
            (root/"episodes.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
            args = SimpleNamespace(model=str(root), data=str(root/"episodes.jsonl"), features=str(root/"features"),
                                   output=str(root/"paired"), shard=0, shards=1, seed=134, rank=2, lr=.001,
                                   margin=.1, epochs=1, smoke=True, objective="paired")
            extract(args, model, rows)
            train(args, model, rows)
            self.assertTrue(all(torch.equal(original[k], v) for k, v in model.state_dict().items()))
            self.assertTrue(all(p.grad is None for p in model.parameters()))
            selected = json.loads((root/"paired/selection.json").read_text())
            self.assertEqual(selected["optimizer_steps"], 2)
            self.assertFalse(selected["passed_dev_gate"])
            args.paired_dir, args.nll_dir, args.output = str(root/"paired"), str(root/"nll"), str(root/"test")
            with self.assertRaisesRegex(ValueError, "test must remain unscored"):
                evaluate_test(args, model, rows)
            self.assertFalse((root/"test").exists())
            # Successful gate is synthetic here; it checks both frozen ablation paths only.
            args.output, args.objective = str(root/"nll"), "nll"
            train(args, model, rows)
            for name in ("paired", "nll"):
                path = root/name/"selection.json"
                value = json.loads(path.read_text())
                value.update(smoke=False, passed_dev_gate=(name == "paired"))
                path.write_text(json.dumps(value))
            args.output = str(root/"test")
            evaluate_test(args, model, rows)
            self.assertEqual(set(json.loads((root/"test/summary.json").read_text())) & {"paired", "nll"}, {"paired", "nll"})


if __name__ == "__main__":
    unittest.main()
