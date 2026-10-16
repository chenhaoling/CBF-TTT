import unittest

from tasks.native_layerwise_write_gate_v2 import ALL_ACTIONS, LAYERS, safe_singleton


def outcome(J, protected=1.0):
    return {"J": J, "protected_accuracy": protected, "protected_nll": J, "eos_rate": 0.0}


class LayerwiseGateTests(unittest.TestCase):
    def test_frozen_action_matrix(self):
        self.assertEqual(len(LAYERS), 7)
        self.assertEqual(len(ALL_ACTIONS), 9)
        self.assertEqual(ALL_ACTIONS[0], "reject_all")
        self.assertEqual(ALL_ACTIONS[-1], "admit_all")

    def test_safe_oracle_rejects_collateral_and_ties(self):
        actions = {name: outcome(2.0) for name in ALL_ACTIONS}
        actions["reject_all"] = outcome(1.0, 1.0)
        actions["admit_layer_0"] = outcome(.5, 0.0)
        self.assertEqual(safe_singleton({"actions": actions}), "reject_all")
        actions["admit_layer_6"] = outcome(.75, 1.0)
        self.assertEqual(safe_singleton({"actions": actions}), "admit_layer_6")


if __name__ == "__main__": unittest.main()
