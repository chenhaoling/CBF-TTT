import unittest

import torch

from cbf_ttt.writer import memory_choice_nll
from scripts.diagnose_cbf_writer_path import amplified_pair, pair_metrics, read_choices


class WriterPathTests(unittest.TestCase):
    def test_amplification_preserves_identity_and_shared_control(self):
        a, b = {0: torch.tensor([[2., 1.]])}, {0: torch.tensor([[.1, -.1]])}
        left, right = amplified_pair(a, b, a, b, 1)
        self.assertIs(left, a)
        self.assertIs(right, b)
        left, right = amplified_pair(a, b, a, b, 0)
        self.assertTrue(torch.equal(left[0], right[0]))
        for scale in (0, 8, 32):
            left, right = amplified_pair(a, b, a, b, scale)
            self.assertLessEqual(float(left[0].norm()), float(a[0].norm())+1e-6)
            self.assertLessEqual(float(right[0].norm()), float(b[0].norm())+1e-6)
        score = pair_metrics([1., 3.], [3., 1.], 0, 1)
        self.assertEqual(score['accuracy'], 1.)
        self.assertEqual(score['matched_gain_vs_twin'], 2.)
        self.assertEqual(pair_metrics([1., 3.], [1., 3.], 0, 1)['matched_gain_vs_twin'], 0.)

    def test_native_and_head_precision_forward(self):
        from inference_model.hf_qwen3.configuration_qwen3 import Qwen3Config
        from inference_model.hf_qwen3.modeling_qwen3 import Qwen3ForCausalLM
        torch.manual_seed(134)
        config = Qwen3Config(vocab_size=32, hidden_size=8, intermediate_size=16,
                            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                            head_dim=4, max_position_embeddings=32, ttt_mode=True,
                            ttt_layers=[0, 1], ttt_chunk=4, ttt_lr=.1, ttt_proj=True)
        model = Qwen3ForCausalLM(config).eval().requires_grad_(False)
        memory = {0: torch.randn(8, 16)*.01, 1: torch.randn(8, 16)*.01}
        for dtype in (torch.float32, torch.bfloat16):
            model.to(dtype)
            expected = memory_choice_nll(model, memory, [3, 4], [5, 6])
            values, hidden = read_choices(model, memory, [3, 4], [5, 6], model.lm_head.weight.float())
            self.assertTrue(torch.allclose(expected.cpu(), torch.tensor(values['native']), atol=1e-6))
            self.assertTrue(torch.isfinite(torch.tensor(values['head_fp32'])).all())
            self.assertEqual(hidden.shape, (1, 1, 8))
        self.assertTrue(all(p.grad is None for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
