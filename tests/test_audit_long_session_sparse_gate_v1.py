import unittest

from scripts.audit_long_session_sparse_gate_v1 import selected_controller, selected_cosine


def metric(pair, update, current=.99, non_target=.99, stale=.9):
    return {
        "pair_state_accuracy": pair,
        "stale_deletion": stale,
        "current_retention": current,
        "non_target_retention": non_target,
        "by_event_kind": {"primary_update": {"accuracy": update}},
    }


class SparseGateAuditTest(unittest.TestCase):
    def test_controller_selection_prefers_feasible_candidate(self):
        grid = {
            "10": {"0.5": metric(.8, .9)},
            "25": {"0.5": metric(.95, .95, non_target=.9)},
        }
        self.assertEqual(selected_controller(grid), (10, .5))

    def test_cosine_selection_reproduces_constraint_order(self):
        grid = {"0.3": metric(.8, .8), "0.6": metric(.9, .9, current=.9)}
        self.assertEqual(selected_cosine(grid), .3)


if __name__ == "__main__":
    unittest.main()
