"""Tests for score-free V3 failure localization."""

import unittest

from tasks.analyze_qwen_natural_forgetting_online_v3 import decision_class


class OnlineV3DiagnosisTests(unittest.TestCase):
    def row(self, desired, incoming="a", deleted=None):
        return {"desired_source_action": desired, "incoming_chain_id": incoming,
                "deleted": [] if deleted is None else [{"chain_id": deleted}]}

    def test_decision_classes(self):
        self.assertEqual(decision_class(self.row("retain")), "retain_true_negative")
        self.assertEqual(decision_class(self.row("retain", deleted="b")),
                         "retain_false_delete")
        self.assertEqual(decision_class(self.row("local")), "local_missed_delete")
        self.assertEqual(decision_class(self.row("local", deleted="a")),
                         "local_correct_delete")
        self.assertEqual(decision_class(self.row("local", deleted="b")),
                         "local_wrong_top_delete")


if __name__ == "__main__":
    unittest.main()
