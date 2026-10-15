import unittest

from tasks.build_qwen_natural_forgetting_utility_pilot_v1 import (
    UNIQUE_ACTIONS,
    apply_intervention,
    choose_oracle,
    future_trace,
    registered_grid,
    selected_decision_steps,
)


class FakeScorer:
    def encode(self, query, keys, seed):
        return ({"top": 0}, None)


def version(index, stem):
    return {"version": index, "prompt": f"{stem} prompt {index}",
            "rephrase_prompt": f"{stem} rephrase {index}",
            "target": f"{stem} answer {index}"}


def session():
    chains = [
        {"id": "a", "kind": "primary", "versions": [version(0, "a"), version(1, "a")]},
        {"id": "ax", "kind": "auxiliary", "parent_chain_id": "a",
         "versions": [version(0, "ax")]},
        {"id": "b", "kind": "primary", "versions": [version(0, "b"), version(1, "b")]},
        {"id": "bx", "kind": "auxiliary", "parent_chain_id": "b",
         "versions": [version(0, "bx")]},
    ]
    events = [
        {"chain_id": "a", "version": 0, "action": "retain", "kind": "primary_initial"},
        {"chain_id": "ax", "version": 0, "action": "retain", "kind": "hard_retain"},
        {"chain_id": "b", "version": 0, "action": "retain", "kind": "primary_initial"},
        {"chain_id": "bx", "version": 0, "action": "retain", "kind": "hard_retain"},
        {"chain_id": "a", "version": 1, "action": "local", "kind": "primary_update"},
        {"chain_id": "b", "version": 1, "action": "local", "kind": "primary_update"},
    ]
    return {"id": "s", "split": "calibration", "chains": chains,
            "events": events, "candidates": [f"v{i}" for i in range(12)]}


def outcome(correct, nll, mass, unrelated):
    return {"correct": correct, "total_answer_nll": nll,
            "mean_memory_mass": mass, "unrelated_correct": unrelated}


class NaturalForgettingUtilityPilotTest(unittest.TestCase):
    def test_grid_has_fifteen_points_and_eleven_executions(self):
        grid = registered_grid()
        self.assertEqual(len(grid), 15)
        self.assertEqual(len(UNIQUE_ACTIONS), 11)
        self.assertEqual(sum(row["canonical"] == "retain_all" for row in grid), 5)

    def test_sparse_steps_include_all_updates_and_two_anchors(self):
        steps = selected_decision_steps(session())
        self.assertEqual(len(steps), 4)
        self.assertTrue({4, 5} <= set(steps))
        kinds = [session()["events"][step]["kind"] for step in steps]
        self.assertIn("primary_initial", kinds)
        self.assertIn("hard_retain", kinds)

    def test_intervention_delete_and_attenuate(self):
        memory = [
            {"chain_id": "a", "version": 0, "prompt": "a", "strength": 1.0},
            {"chain_id": "b", "version": 0, "prompt": "b", "strength": 1.0},
            {"chain_id": "c", "version": 0, "prompt": "c", "strength": 1.0},
        ]
        deleted, meta = apply_intervention(memory, "last2_a0", "q", FakeScorer())
        self.assertEqual([row["chain_id"] for row in deleted], ["a"])
        self.assertEqual(meta["indices"], [1, 2])
        attenuated, _ = apply_intervention(memory, "hybrid_a05", "q", FakeScorer())
        self.assertEqual([row["strength"] for row in attenuated], [.5, 1., .5])
        self.assertEqual([row["strength"] for row in memory], [1., 1., 1.])

    def test_future_trace_uses_one_intervention_then_retain(self):
        traces, meta = future_trace(session(), 4, "scope_top1_a0", FakeScorer())
        self.assertEqual([row["phase"] for row in traces], ["update_1", "update_2"])
        self.assertEqual(len(meta["indices"]), 1)
        # Four old records - one deletion + current append + future append.
        self.assertEqual([len(row["memory"]) for row in traces], [4, 5])

    def test_control_barrier_excludes_unrelated_harm(self):
        actions = {name: outcome(5, 5., 5., 4) for name in UNIQUE_ACTIONS}
        actions["retain_all"] = outcome(5, 5., 6., 4)
        actions["global_a0"] = outcome(7, 2., 1., 3)  # better, but unsafe
        actions["scope_top1_a0"] = outcome(6, 3., 4., 4)
        self.assertEqual(choose_oracle(actions, constrained=False), "global_a0")
        self.assertEqual(choose_oracle(actions, constrained=True), "scope_top1_a0")


if __name__ == "__main__":
    unittest.main()
