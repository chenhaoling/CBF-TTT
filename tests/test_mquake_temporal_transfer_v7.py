import unittest

from tasks.evaluate_mquake_temporal_transfer_v7 import transfer_gate


class MquakeTemporalTransferV7Test(unittest.TestCase):
    def metric(self):
        return {"current_retention": .98, "non_target_retention": .97,
                "stale_deletion": .72, "pair_state_accuracy": .71,
                "by_event_kind": {"hard_retain": {"accuracy": .96}}}

    def test_gate_requires_transfer_and_safety(self):
        comparison = {"bootstrap_95ci": [.02, .20]}
        self.assertTrue(transfer_gate(self.metric(), .92, .84, comparison)["passed"])
        metric = self.metric(); metric["stale_deletion"] = .69
        self.assertFalse(transfer_gate(metric, .92, .84, comparison)["passed"])

    def test_gate_requires_confident_random_gain(self):
        comparison = {"bootstrap_95ci": [-.01, .20]}
        self.assertFalse(transfer_gate(self.metric(), .92, .84, comparison)["passed"])


if __name__ == "__main__":
    unittest.main()
