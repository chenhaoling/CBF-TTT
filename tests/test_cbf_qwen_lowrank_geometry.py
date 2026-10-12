"""Tests for the registered Qwen low-rank geometry diagnostic."""

import unittest

from tasks.diagnose_cbf_qwen_lowrank_geometry import (
    REGULARIZERS, VARIANT_ORDER, readability_gate, select_regularizer,
)
from tasks.train_cbf_qwen_lowrank_forgetting import SEEDS


class SelectionTests(unittest.TestCase):
    def test_largest_regularizer_within_one_point(self):
        grid = {
            "rp128_whiten_r0.0001": {"mean_accuracy": .94},
            "rp128_whiten_r0.001": {"mean_accuracy": .945},
            "rp128_whiten_r0.01": {"mean_accuracy": .936},
        }
        self.assertEqual(select_regularizer(grid, 128), .01)

    def test_gate_selects_first_passing_variant(self):
        def row(accuracy, family=None):
            family = accuracy if family is None else family
            return {"mean_accuracy": accuracy, "min_seed_accuracy": accuracy,
                    "min_family_accuracy": family}
        dev = {
            "rp128": row(.70), "rp128_whiten": row(.93),
            "rp256_whiten": row(.96), "random128": row(.99),
        }
        sweeps = {name: {str(seed): {"16": {"accuracy": .92}} for seed in SEEDS}
                  for name in VARIANT_ORDER}
        gate = readability_gate(dev, sweeps)
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["selected_variant"], "rp128_whiten")

    def test_registered_grid_is_frozen(self):
        self.assertEqual(REGULARIZERS, (1e-4, 1e-3, 1e-2))
        self.assertEqual(VARIANT_ORDER, ("rp128", "rp128_whiten", "rp256_whiten"))


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch tests run on experiment server")
class GeometryTests(unittest.TestCase):
    def test_whitening_reduces_projected_anisotropy(self):
        from tasks.diagnose_cbf_qwen_lowrank_geometry import geometry, _normalize_rows
        torch.manual_seed(17)
        latent = torch.randn(512, 8)
        transform = torch.randn(8, 32) * torch.linspace(1, 12, 32)
        rows = latent @ transform
        covariance = rows.T @ rows / (rows.shape[0] - 1)
        values, vectors = torch.linalg.eigh(covariance)
        scale = values.clamp_min(0).mean()
        white = (rows @ vectors * torch.rsqrt(values.clamp_min(0) + 1e-3 * scale)) @ vectors.T
        self.assertLess(geometry(_normalize_rows(white))["absolute_off_diagonal_cosine"]["mean"],
                        geometry(_normalize_rows(rows))["absolute_off_diagonal_cosine"]["mean"])

    def test_rp128_is_bit_exact_v1_bridge(self):
        from tasks.diagnose_cbf_qwen_lowrank_geometry import build_representations
        from tasks.train_cbf_qwen_lowrank_forgetting import build_embeddings
        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                return [ord(char) % 31 for char in text]
        class Model:
            def __init__(self):
                self.embedding = torch.nn.Embedding(32, 12)
                self.lm_head = torch.nn.Linear(12, 32, bias=False)
            def get_input_embeddings(self): return self.embedding
        pairs = [({}, {"semantic_events": [[{"op": "assign", "facts": {
            "unit_a": "code_1", "unit_b": "code_2"}}]]})]
        model, tokenizer = Model(), Tokenizer()
        _, _, meta = build_representations(model, tokenizer, pairs)
        _, old_meta, _ = build_embeddings(model, tokenizer, pairs, 128)
        self.assertEqual(meta["hashes"]["rp128"], old_meta["hashes"])


if __name__ == "__main__":
    unittest.main()
