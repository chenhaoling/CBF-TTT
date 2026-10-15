import importlib.util
import unittest

from tasks.fit_retrieval_evidence_forgetting_s3_fusion import (
    conformal_negative_threshold,
    delete_counts,
    empirical_threshold,
    negative_js,
    reciprocal_overlap,
)


class RetrievalEvidenceForgettingS3FusionTests(unittest.TestCase):
    def test_delete_counts_follow_five_scenario_rule(self):
        value = delete_counts([True, True, False, False], [True, False, True, False])
        self.assertEqual((value["tp"], value["fn"], value["fp"], value["tn"]),
                         (2, 2, 1, 5))

    def test_empirical_threshold_enforces_precision(self):
        value = empirical_threshold([.9, .8, .7, .6], [True, False, True, False])
        self.assertEqual(value["threshold"], .9)
        self.assertEqual(value["precision"], 1.0)

    def test_conformal_threshold_is_strictly_above_selected_negative(self):
        value = conformal_negative_threshold(
            [.9, .8, .7, .6], [False, True, False, True], ["a", "a", "b", "b"], .5
        )
        self.assertGreater(value, .7)
        self.assertLess(value, .91)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy is exercised remotely")
    def test_support_overlap_and_js_are_oriented(self):
        import numpy as np

        self.assertGreater(reciprocal_overlap(list(range(16)), list(range(16))), .99)
        same = negative_js(np.asarray([.8, .2]), np.asarray([.8, .2]))
        different = negative_js(np.asarray([1., 0.]), np.asarray([0., 1.]))
        self.assertEqual(same, 0.0)
        self.assertLess(different, same)


if __name__ == "__main__":
    unittest.main()
