import unittest

from scripts.audit_factorized_causal_forgetting_v4 import selected_gate


def metric(pair, update, current=.99, non_target=.98, hard=.98):
    return {"pair_state_accuracy": pair, "current_retention": current,
            "non_target_retention": non_target,
            "by_event_kind": {"hard_retain": {"accuracy": hard},
                              "primary_update": {"accuracy": update}}}


class FactorizedCausalAuditTest(unittest.TestCase):
    def test_safe_candidate_beats_higher_unsafe_pair_score(self):
        grid = {"10": {"0.9": metric(.9, .9)},
                "25": {"0.9": metric(.95, .95, non_target=.9)}}
        self.assertEqual(selected_gate(grid), (10, .9))


if __name__ == "__main__": unittest.main()
