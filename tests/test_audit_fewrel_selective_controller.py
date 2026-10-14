import unittest

from scripts.audit_fewrel_selective_controller import close_tree, complete_metric


class AuditFewRelSelectiveControllerTest(unittest.TestCase):
    def test_metric_and_tree_reconstruction(self):
        value = complete_metric([1, 1, 0, 0], [1, 0, 1, 0], [.9, .4, .6, .1])
        self.assertEqual(value["balanced_accuracy"], .5)
        self.assertEqual(value["auroc"], .75)
        self.assertEqual(value["false_deletion_rate"], .5)
        self.assertTrue(close_tree({"x": 1}, {"x": 1.0}))
        self.assertFalse(close_tree({"x": 1}, {"x": 2}))


if __name__ == "__main__":
    unittest.main()
