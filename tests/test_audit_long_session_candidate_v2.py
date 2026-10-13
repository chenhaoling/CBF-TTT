import unittest

from scripts.audit_long_session_candidate_v2 import selected_epoch


class CandidateV2AuditTest(unittest.TestCase):
    def test_epoch_selection_uses_worst_length_before_mrr(self):
        grid = {
            "10": {"top1": .9, "mrr": .91,
                   "by_length": {"16": {"top1": .8}, "32": {"top1": .9}}},
            "25": {"top1": .9, "mrr": .99,
                   "by_length": {"16": {"top1": .7}, "32": {"top1": 1.0}}},
        }
        self.assertEqual(selected_epoch(grid), 10)


if __name__ == "__main__": unittest.main()
