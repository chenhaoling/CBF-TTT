import unittest

from tasks.train_mquake_temporal_adaptation_v8 import development_gate, tune_gate


def metric(pair=.82):
    return {"current_retention": .98, "non_target_retention": .97,
            "stale_deletion": .84, "pair_state_accuracy": pair,
            "by_event_kind": {"hard_retain": {"accuracy": .96}},
            "by_length": {str(length): {"pair_state_accuracy": .75}
                          for length in (16, 32, 48, 64)}}


class MquakeTemporalAdaptationV8Test(unittest.TestCase):
    def test_tune_gate(self):
        candidate = {"shortlist_recall": .92, "candidate_top1": .84}
        self.assertTrue(tune_gate(metric(), candidate)["passed"])
        candidate["candidate_top1"] = .79
        self.assertFalse(tune_gate(metric(), candidate)["passed"])

    def test_development_gate(self):
        candidate = {"shortlist_recall": .92, "candidate_top1": .88}
        comparison = {"bootstrap_95ci": [.03, .15]}
        self.assertTrue(development_gate(metric(), candidate, comparison)["passed"])
        comparison["bootstrap_95ci"][0] = -.01
        self.assertFalse(development_gate(metric(), candidate, comparison)["passed"])


if __name__ == "__main__":
    unittest.main()
