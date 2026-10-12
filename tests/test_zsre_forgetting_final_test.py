"""Tests for the one-time held-out forgetting test release."""

import unittest

from tasks.evaluate_zsre_forgetting_final_test import release_threshold


class FinalTestReleaseTests(unittest.TestCase):
    @staticmethod
    def sources():
        summary = {"test_loaded": False, "test_scored": False,
                   "selected_passing_original_gate": {
                       "threshold": .675, "development_gate": {"passed": True}}}
        manifest = {"test_loaded": False, "test_scored": False}
        audit = {"audit_passed": True}
        return summary, manifest, audit

    def test_release_requires_audited_passing_boundary(self):
        summary, manifest, audit = self.sources()
        self.assertEqual(release_threshold(summary, manifest, audit), .675)
        audit["audit_passed"] = False
        with self.assertRaises(ValueError):
            release_threshold(summary, manifest, audit)

    def test_release_rejects_prior_test_access(self):
        summary, manifest, audit = self.sources(); summary["test_loaded"] = True
        with self.assertRaises(ValueError):
            release_threshold(summary, manifest, audit)


if __name__ == "__main__":
    unittest.main()
