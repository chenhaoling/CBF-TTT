import unittest

from tasks.train_native_eos_safe_layerwise_write_gate_v4 import (
    build_model,
    choose_actions,
    feature_indices,
    predict,
)


class NativeEosSafeLayerwiseControllerV4Test(unittest.TestCase):
    def test_feature_ablations_are_disjoint_and_complete(self):
        no_support = feature_indices("no_support")
        support_only = feature_indices("support_only")
        full = feature_indices("full")
        self.assertEqual(len(full), 1854)
        self.assertEqual(len(no_support), 41)
        self.assertEqual(len(support_only), 1815)
        self.assertEqual(set(no_support) | set(support_only), set(full))
        self.assertEqual(set(no_support) & set(support_only), {1852, 1853})

    def test_four_head_output_and_eos_veto(self):
        import torch

        model = build_model(41)
        x = torch.zeros(2, 41)
        stats = {
            "x_mean": torch.zeros(41), "x_std": torch.ones(41),
            "benefit_mean": torch.zeros(7), "benefit_std": torch.ones(7),
            "delta_mean": torch.zeros(7), "delta_std": torch.ones(7),
        }
        outputs = predict(model, x, stats)
        self.assertEqual(len(outputs), 4)
        self.assertTrue(all(value.shape == (2, 7) for value in outputs))
        benefit = torch.ones(1, 7)
        delta = torch.zeros(1, 7)
        safe = torch.zeros(1, 7)
        eos_harm = torch.ones(1, 7)
        self.assertEqual(choose_actions(benefit, delta, safe, eos_harm), ["reject_all"])


if __name__ == "__main__":
    unittest.main()
