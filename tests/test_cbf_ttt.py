"""Small tensor checks for the optional CBF update and controller regression."""

import json
import math
import tempfile
from types import SimpleNamespace
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from cbf_ttt.controller import ForgettingController
from cbf_ttt.runtime import CBFSession, cbf_forward_mlp, token_statistics


class TinyMLP(nn.Module):
    def __init__(self, projection: bool):
        super().__init__()
        self.hidden_size = 4
        self.ttt_chunk = 3
        self.ttt_lr = 0.2
        self.gate_proj = nn.Linear(4, 6)
        self.up_proj = nn.Linear(4, 6)
        self.down_proj = nn.Linear(6, 4)
        self.ttt_conv = nn.Conv1d(4, 4, 5, padding=2, groups=4, bias=False)
        self.ttt_proj = nn.Linear(4, 4, bias=False) if projection else None
        self.act_fn = F.silu


class CBFCoreTests(unittest.TestCase):
    def test_staged_candidate_preserves_current_output_and_bias(self):
        torch.manual_seed(7)
        x = torch.randn(1, 3, 4)
        target = torch.randn(1, 3, 4)
        for projection in (False, True):
            mlp = TinyMLP(projection)
            memory = torch.randn_like(mlp.down_proj.weight) * 0.01
            cache = SimpleNamespace(cbf_memory={0: memory}, cbf_candidates={}, cbf_collect=True)
            output = cbf_forward_mlp(mlp, x, target, cache, 0)
            h = mlp.act_fn(mlp.gate_proj(x)) * mlp.up_proj(x)
            self.assertTrue(torch.allclose(output, F.linear(h, mlp.down_proj.weight + memory, mlp.down_proj.bias)))
            conv = mlp.ttt_conv(target.transpose(1, 2)).transpose(1, 2)
            expected = conv[0].T @ h[0]
            if projection:
                expected = mlp.ttt_proj.weight.T @ expected
            self.assertEqual(tuple(cache.cbf_candidates[0].shape), tuple(mlp.down_proj.weight.shape))
            self.assertTrue(torch.allclose(cache.cbf_candidates[0], expected * mlp.ttt_lr))
            cache.cbf_collect = False
            cache.cbf_candidates = {}
            cbf_forward_mlp(mlp, x[:, :1], target[:, :1], cache, 0)
            self.assertEqual(cache.cbf_candidates, {})

    def test_controller_has_gradient_outside_interval(self):
        controller = ForgettingController(4, 7, width=8, semantic_size=4)
        with torch.no_grad():
            controller.regressor[-1].weight.zero_()
            controller.regressor[-1].bias.fill_(2.0)
        raw = controller(torch.zeros(2, 4), torch.zeros(2, 7))
        F.mse_loss(raw, torch.zeros_like(raw)).backward()
        self.assertGreater(controller.regressor[-1].bias.grad.abs().item(), 0)
        self.assertTrue(torch.equal(controller.predict(torch.zeros(2, 4), torch.zeros(2, 7)), torch.ones(2)))

    def test_commit_forgets_only_old_memory(self):
        for alpha in (0.0, 0.5, 1.0):
            old = torch.tensor([[2.0]])
            candidate = torch.tensor([[3.0]])
            cache = SimpleNamespace(
                cbf_memory={0: old, 2: old * 2},
                cbf_candidates={0: candidate, 2: candidate * 2},
                cbf_collect=True,
            )
            fake_session = SimpleNamespace(cache=cache, layers=(0, 2))
            CBFSession.commit(fake_session, alpha)
            self.assertEqual(cache.cbf_memory[0].item(), (1.0 - alpha) * 2.0 + 3.0)
            self.assertEqual(cache.cbf_memory[2].item(), (1.0 - alpha) * 4.0 + 6.0)
            self.assertEqual(cache.cbf_candidates, {})
            self.assertFalse(cache.cbf_collect)

    def test_lexical_statistics(self):
        stats = token_statistics(torch.tensor([1, 2, 1, 2]))
        self.assertAlmostEqual(stats[0].item(), 0.693147, places=5)
        self.assertAlmostEqual(stats[1].item(), 0.5, places=5)
        self.assertAlmostEqual(stats[2].item(), 1 / 3, places=5)

    def test_tiny_qwen_and_llama_session(self):
        from cbf_ttt.experiment import collect_labels, summarize_label_profile
        from inference_model.hf_llama3.configuration_llama import LlamaConfig
        from inference_model.hf_llama3.modeling_llama import LlamaForCausalLM
        from inference_model.hf_qwen3.configuration_qwen3 import Qwen3Config
        from inference_model.hf_qwen3.modeling_qwen3 import Qwen3ForCausalLM

        common = dict(
            vocab_size=32, hidden_size=8, intermediate_size=16, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=2, head_dim=4,
            max_position_embeddings=32, ttt_mode=True, ttt_layers=[0, 1],
            ttt_chunk=4, ttt_lr=0.1,
        )
        for model_class, config_class, extra in (
            (Qwen3ForCausalLM, Qwen3Config, {"ttt_proj": True}),
            (LlamaForCausalLM, LlamaConfig, {"ttt_proj": False, "mlp_bias": True}),
        ):
            model = model_class(config_class(**common, **extra)).eval()
            model.requires_grad_(False)
            session = CBFSession(model)
            semantic, scalars = session.observe([1, 2, 3, 4])
            self.assertEqual(tuple(semantic.shape), (1, 8))
            self.assertEqual(tuple(scalars.shape), (1, 10))
            session.commit(0.5)
            for layer_idx in (0, 1):
                self.assertEqual(
                    tuple(session.cache.cbf_memory[layer_idx].shape),
                    tuple(model.model.layers[layer_idx].mlp.down_proj.weight.shape),
                )
            session.consume([5, 6], "baseline")
            length_before = session.cache.get_seq_length()
            self.assertTrue(math.isfinite(session.score_answer([7, 8], [9, 10])))
            self.assertEqual(session.cache.get_seq_length(), length_before)
            scenario = {
                "id": "tiny", "group_id": "tiny-group", "split": "train",
                "context_ids": [1, 2, 3, 4, 5, 6, 7, 8],
                "futures": [{"continuation_ids": [9], "queries": [{"query_ids": [10], "answer_ids": [11]}]}],
            }
            with tempfile.TemporaryDirectory() as directory:
                path = str(Path(directory) / "labels.jsonl")
                self.assertEqual(collect_labels(model, [scenario], path, [0.0, 0.5, 1.0], every=2), 1)
                row = json.loads(Path(path).read_text())
                self.assertEqual(row["grid"], [0.0, 0.5, 1.0])
                self.assertEqual(len(row["losses"]), 3)
                self.assertIn(row["alpha_star"], row["grid"])
                self.assertGreater(row["label_time_s"], 0)
                self.assertIsNone(row["peak_allocated_gib"])
                self.assertEqual(summarize_label_profile(path)["labels"], 1)

    def test_alpha_zero_matches_qwen_baseline_at_same_chunk_boundaries(self):
        from inference_model.hf_qwen3.configuration_qwen3 import Qwen3Config
        from inference_model.hf_qwen3.modeling_qwen3 import Qwen3ForCausalLM, TTTDynamicCache

        config = Qwen3Config(
            vocab_size=32, hidden_size=8, intermediate_size=16, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=2, head_dim=4,
            max_position_embeddings=32, ttt_mode=True, ttt_layers=[0, 1],
            ttt_chunk=4, ttt_lr=0.1,
        )
        model = Qwen3ForCausalLM(config).eval()
        model.requires_grad_(False)
        standard_cache = TTTDynamicCache(config=config)
        controlled = CBFSession(model)
        with torch.inference_mode():
            for ids in ([1, 2, 3, 4], [5, 6, 7, 8]):
                model.model(input_ids=torch.tensor([ids]), past_key_values=standard_cache, use_cache=True)
                controlled.step(ids, "baseline")
            standard_query = model.model(
                input_ids=torch.tensor([[9, 10]]), past_key_values=standard_cache, use_cache=True
            ).last_hidden_state
            controlled_query = controlled._forward([9, 10])
        self.assertTrue(torch.allclose(standard_query, controlled_query, atol=1e-5, rtol=1e-4))


if __name__ == "__main__":
    unittest.main()
