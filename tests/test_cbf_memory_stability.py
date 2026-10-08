"""Validate native bridging, precision controls, and read-only diagnostic state."""

import unittest
from unittest.mock import patch

import torch

from cbf_ttt.runtime import cbf_forward_mlp
from scripts.diagnose_cbf_memory_stability import (
    candidate_probe, cbf_delta, commit, diagnostic_forward, effective, new_session, tensor_digest,
)


class StabilityTests(unittest.TestCase):
    def model(self, dtype=torch.float32):
        from inference_model.hf_qwen3.configuration_qwen3 import Qwen3Config
        from inference_model.hf_qwen3.modeling_qwen3 import Qwen3ForCausalLM
        torch.manual_seed(108)
        config = Qwen3Config(vocab_size=32, hidden_size=8, intermediate_size=16,
                             num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                             head_dim=4, max_position_embeddings=64, ttt_mode=True,
                             ttt_layers=[0, 1], ttt_chunk=4, ttt_lr=.1, ttt_proj=True)
        model = Qwen3ForCausalLM(config).to(dtype).eval().requires_grad_(False)
        with torch.no_grad():
            for layer in model.model.layers:
                layer.mlp.ttt_conv.weight.normal_(0, .02)
        return model

    def test_native_bridge_matches_hidden_weights_and_nll(self):
        for dtype in (torch.float32, torch.bfloat16):
            model = self.model(dtype)
            native, bridge = new_session(model, "native"), new_session(model, "native_ops_w")
            with patch("inference_model.hf_qwen3.modeling_qwen3.cbf_forward_mlp", diagnostic_forward):
                for _ in range(3):
                    a = native._forward([1, 2, 3, 4])
                    b = bridge._forward([1, 2, 3, 4], collect=True)
                    commit(bridge)
                    self.assertTrue(torch.equal(a, b))
                    for layer in bridge.layers:
                        self.assertTrue(torch.equal(effective(model, native.cache, layer),
                                                    effective(model, bridge.cache, layer)))
                    self.assertEqual(native.score_answer([5], [6, 7]), bridge.score_answer([5], [6, 7]))

    def test_accumulator_precision_isolated_from_backbone(self):
        model = self.model(torch.bfloat16)
        with torch.no_grad():
            for layer in model.model.layers:
                layer.mlp.down_proj.weight.fill_(1.)
        states = {mode: new_session(model, mode) for mode in
                  ("native_ops_w", "native_ops_m", "native_ops_w_fp32", "native_ops_m_fp32")}
        for _ in range(3):
            for session in states.values():
                session.cache.cbf_candidates = {i: torch.full_like(model.model.layers[i].mlp.down_proj.weight, .003)
                                                for i in session.layers}
                commit(session)
        w = effective(model, states["native_ops_w"].cache, 0)
        m = effective(model, states["native_ops_m"].cache, 0)
        self.assertFalse(torch.equal(w, m))
        self.assertTrue(torch.equal(effective(model, states["native_ops_w_fp32"].cache, 0),
                                    effective(model, states["native_ops_m_fp32"].cache, 0)))
        self.assertEqual(states["native_ops_m_fp32"].cache.cbf_memory[0].dtype, torch.float32)
        self.assertTrue(all(p.dtype == torch.bfloat16 and not p.requires_grad for p in model.parameters()))

    def test_query_clone_and_no_update_reference(self):
        model = self.model()
        with patch("inference_model.hf_qwen3.modeling_qwen3.cbf_forward_mlp", diagnostic_forward):
            for mode in ("native", "cbf", "none", "native_ops_w_fp32", "native_ops_m_fp32", "fp32_m"):
                session = new_session(model, mode)
                session._forward([1, 2, 3, 4], collect=mode != "native")
                commit(session)
                hashes = {i: tensor_digest(effective(model, session.cache, i)) for i in session.layers}
                seq = session.cache.get_seq_length()
                session.score_answer([5], [6, 7])
                self.assertEqual(seq, session.cache.get_seq_length())
                self.assertEqual(hashes, {i: tensor_digest(effective(model, session.cache, i)) for i in session.layers})
                if mode == "none":
                    for i in session.layers:
                        self.assertTrue(torch.equal(effective(model, session.cache, i),
                                                    model.model.layers[i].mlp.down_proj.weight))
        import inference_model.hf_qwen3.modeling_qwen3 as module
        self.assertIs(module.cbf_forward_mlp, cbf_forward_mlp)

    def test_same_input_candidate_reference(self):
        model = self.model()
        mlp = model.model.layers[0].mlp
        h, conv = torch.randn(1, 4, 16), torch.randn(1, 4, 8)
        staged = cbf_delta(mlp, h, conv)
        values = candidate_probe(mlp, h, conv, staged)
        self.assertEqual(values["cbf_relative_error_vs_fp32"], 0.)
        self.assertLess(values["native_relative_error_vs_fp32"], 1e-5)


if __name__ == "__main__":
    unittest.main()
