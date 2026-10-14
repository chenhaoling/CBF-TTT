"""Pure unit tests for frozen V16 formal aggregation and state reconstruction."""

import unittest

from tasks.build_mquake_qwen_outcome_v16_formal_states import POLICIES
from tasks.evaluate_mquake_qwen_outcome_v16_formal import (
    aggregate_formal, clustered_interval, memory_from_state, select_fixed_by_outer_fold,
)


KINDS = ("primary_old_phrase", "primary_current_phrase", "hard_retain", "unrelated")


def query(kind, correct, nll):
    return {
        "kind": kind, "correct": correct, "answer_nll": nll,
        "prompt_tokens": 100, "total_tokens_max": 102,
        "generation": {"exact": correct, "normalized_exact": correct},
        "resources": {"seconds": .1, "gpu_peak_allocated_gib": 8.0,
                      "gpu_peak_reserved_gib": 8.5},
    }


def row(index, v15_correct=1, fixed_correct=0):
    policies = {}
    for policy in POLICIES:
        correct = v15_correct if policy in ("v15", "oracle") else fixed_correct
        nll = .1 if correct else 1.0
        policies[policy] = {
            "memory_session_ids": [] if policy == "no_memory" else [f"s{index}"],
            "score_source_policy": policy,
            "queries": [query(kind, correct, nll) for kind in KINDS],
        }
    return {"episode_id": f"e{index}", "subject": f"subject-{index // 4}",
            "fold": (index // 4) % 5, "length": (16, 32, 48, 64)[index % 4],
            "policies": policies}


class FormalOutcomeTest(unittest.TestCase):
    def test_memory_reconstruction_rejects_future(self):
        episode = {"sessions": [{"session_id": "a"}, {"session_id": "b"},
                                {"session_id": "future"}]}
        sample = {"query_step": 1}
        self.assertEqual([value["session_id"] for value in
                          memory_from_state(episode, sample, ["b", "a"])], ["b", "a"])
        with self.assertRaises(ValueError):
            memory_from_state(episode, sample, ["future"])

    def test_cluster_bootstrap_is_subject_level_and_deterministic(self):
        records = [{"subject": "a", "gain": 1.0}, {"subject": "a", "gain": 1.0},
                   {"subject": "b", "gain": 3.0}]
        first = clustered_interval(records, "gain", draws=1000, seed=9)
        second = clustered_interval(records, "gain", draws=1000, seed=9)
        self.assertEqual(first, second)
        self.assertEqual(first["mean"], 2.0)
        self.assertEqual(first["subjects"], 2)

    def test_fixed_selection_never_reads_heldout_fold(self):
        rows = [row(index) for index in range(20)]
        for value in rows:
            if value["fold"] == 0:
                for query_value in value["policies"]["retain_all"]["queries"]:
                    query_value["correct"] = 1
                    query_value["answer_nll"] = 0.0
        selected, _ = select_fixed_by_outer_fold(rows)
        self.assertEqual(selected["0"], "no_memory")

    def test_registered_gate_and_nine_policy_table(self):
        rows = [row(index) for index in range(272)]
        result = aggregate_formal(rows)
        self.assertEqual(set(result["policies"]), set(POLICIES))
        self.assertEqual(result["fixed_selection"]["selected"],
                         {str(fold): "no_memory" for fold in range(5)})
        self.assertTrue(result["outcome_gate"]["passed"])
        self.assertEqual(result["policies"]["v15"]["primary_pair_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
