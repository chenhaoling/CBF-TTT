"""Nonzero-update parity and causal gradient coverage in repository training code."""

import unittest

import torch

from scripts.diagnose_ttt_train_infer import hidden_forward, update_scale


class TrainInferTests(unittest.TestCase):
    def pair(self):
        from hf_models.hf_qwen3.configuration_qwen3 import Qwen3Config
        from hf_models.hf_qwen3.modeling_qwen3 import Qwen3ForCausalLM as TrainModel
        from inference_model.hf_qwen3.configuration_qwen3 import Qwen3Config as NativeConfig
        from inference_model.hf_qwen3.modeling_qwen3 import Qwen3ForCausalLM as NativeModel
        torch.manual_seed(108)
        config = Qwen3Config(vocab_size=32, hidden_size=8, intermediate_size=16,
                            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                            head_dim=4, max_position_embeddings=64, ttt_mode=True,
                            ttt_layers=[0, 1], ttt_chunk=4, ttt_lr=.1, ttt_proj=True)
        config._attn_implementation = "eager"
        train = TrainModel(config).eval()
        with torch.no_grad():
            # Exercise real updates; zero conv initialization would make parity vacuous.
            for layer in train.model.layers:
                layer.mlp.ttt_conv.weight.normal_(0, .05)
        native_config = NativeConfig(**config.to_dict())
        native_config._attn_implementation = "eager"
        native = NativeModel(native_config).eval()
        native.load_state_dict(train.state_dict(), strict=True)
        return train, native

    def test_fp32_full_and_streaming_partial_complete_chunks(self):
        train, native = self.pair()
        for length in (6, 8, 12, 14):
            ids = torch.arange(length).unsqueeze(0)%32
            with torch.no_grad():
                a, b, c = hidden_forward(train, ids), hidden_forward(native, ids), hidden_forward(native, ids, True)
            torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(a, c, atol=2e-5, rtol=2e-5)

    def test_writer_receives_gradients_from_later_chunk_only(self):
        train, _ = self.pair()
        mlp = train.model.layers[0].mlp
        x = torch.randn(1, 12, 8, requires_grad=True)
        t = torch.randn(1, 12, 8, requires_grad=True)
        output = mlp(x, t)
        output[:, :4].square().sum().backward(retain_graph=True)
        self.assertEqual(float(t.grad.abs().sum()), 0.)
        t.grad.zero_()
        output[:, 8:12].square().sum().backward()
        self.assertGreater(float(t.grad[:, :8].abs().sum()), 0.)
        self.assertEqual(float(t.grad[:, 8:].abs().sum()), 0.)
        for p in (mlp.ttt_conv.weight, mlp.ttt_proj.weight):
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(float(p.grad.abs().sum()), 0.)

    def test_zero_update_control_and_scale_restoration(self):
        train, native = self.pair()
        ids = torch.arange(12).unsqueeze(0)
        with torch.no_grad(), update_scale(train, False), update_scale(native, False):
            a, b = hidden_forward(train, ids), hidden_forward(native, ids, True)
            torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
        self.assertEqual(train.model.layers[0].mlp.ttt_lr, .1)
        self.assertEqual(native.model.layers[0].mlp.ttt_lr, .1)
        try:
            with update_scale(train, False):
                raise RuntimeError("test")
        except RuntimeError:
            pass
        self.assertEqual(train.model.layers[0].mlp.ttt_lr, .1)


if __name__ == "__main__":
    unittest.main()
