import unittest

from tasks.train_native_eos_safe_layerwise_write_gate_v4 import feature_indices


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


if __name__ == "__main__":
    unittest.main()
