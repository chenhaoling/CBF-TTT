import unittest

from tasks.train_mquake_anchor_forgetting_v10 import development_gate, tune_gate


def metric(pair=.88):
    return {"current_retention": .98, "non_target_retention": .97,
            "stale_deletion": .89, "pair_state_accuracy": pair,
            "by_event_kind": {"hard_retain": {"accuracy": .96}},
            "by_length": {str(length): {"pair_state_accuracy": .83}
                          for length in (16, 32, 48, 64)}}


class MquakeAnchorForgettingV10Test(unittest.TestCase):
    def test_tune_gate(self):
        candidate = {"shortlist_recall": 1., "candidate_top1": 1.}
        self.assertTrue(tune_gate(metric(), candidate)["passed"])
        candidate["candidate_top1"] = .98
        self.assertFalse(tune_gate(metric(), candidate)["passed"])

    def test_development_gate(self):
        candidate = {"shortlist_recall": 1., "candidate_top1": 1.}
        comparison = {"bootstrap_95ci": [.02, .15]}
        self.assertTrue(development_gate(metric(), candidate, comparison)["passed"])
        comparison["bootstrap_95ci"][0] = 0.
        self.assertFalse(development_gate(metric(), candidate, comparison)["passed"])


if __name__ == "__main__": unittest.main()
