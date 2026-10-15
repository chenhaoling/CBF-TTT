import unittest

from tasks.build_qwen_natural_causal_memory_v1 import session_candidates
from tasks.evaluate_qwen_natural_causal_memory_v1 import (
    ARMS,
    STATE_POLICIES,
    advance,
    aggregate,
    checkpoint_label,
)


def chain(identifier, kind, targets, parent=None):
    row = {
        "id": identifier,
        "subject": f"subject-{identifier}",
        "kind": kind,
        "versions": [
            {"version": index, "prompt": f"prompt-{identifier}-{index}",
             "rephrase_prompt": f"rephrase-{identifier}-{index}", "target": target}
            for index, target in enumerate(targets)
        ],
    }
    if parent is not None:
        row["parent_chain_id"] = parent
    return row


def partial_session():
    chains = [
        chain("p0", "primary", ["old zero", "new zero"]),
        chain("a0", "auxiliary", ["aux zero"], "p0"),
        chain("p1", "primary", ["old one", "new one"]),
        chain("a1", "auxiliary", ["aux one"], "p1"),
    ]
    events = [
        {"chain_id": "p0", "version": 0, "action": "retain", "kind": "primary_initial"},
        {"chain_id": "a0", "version": 0, "action": "retain", "kind": "hard_retain"},
        {"chain_id": "p1", "version": 0, "action": "retain", "kind": "primary_initial"},
        {"chain_id": "a1", "version": 0, "action": "retain", "kind": "hard_retain"},
        {"chain_id": "p0", "version": 1, "action": "local", "kind": "primary_update"},
        {"chain_id": "p1", "version": 1, "action": "local", "kind": "primary_update"},
    ]
    return {"id": "session", "split": "calibration", "chains": chains, "events": events}


class DummyController:
    def predict(self, query, keys):
        return {"prediction": 0 if keys and query.startswith("prompt-p0-1") else len(keys),
                "probability": .9}


class DummyScorer:
    def score(self, query, keys, seed):
        return {"top": 0, "top_score": .9, "cosine_retrieve": True}


class NaturalCausalMemoryTest(unittest.TestCase):
    def test_candidates_are_deterministic_unique_and_complete(self):
        session = partial_session()
        pool = [f"pool {index}" for index in range(30)]
        left = session_candidates(session, pool)
        right = session_candidates(session, list(reversed(pool)))
        self.assertEqual(left, right)
        self.assertEqual(len(left), 12)
        self.assertEqual(len({value.casefold() for value in left}), 12)
        for value in ("old zero", "new zero", "aux zero", "old one", "new one", "aux one"):
            self.assertIn(value, left)

    def test_partial_session_checkpoint_schedule(self):
        session = partial_session()
        self.assertEqual(
            [checkpoint_label(index, session) for index in range(6)],
            [None, None, None, "initial", "update_1", "update_2"],
        )

    def test_oracle_local_update_removes_only_old_chain(self):
        session = partial_session(); chains = {row["id"]: row for row in session["chains"]}
        memory = []
        for event in session["events"]:
            version = chains[event["chain_id"]]["versions"][event["version"]]
            memory, decision = advance(
                "oracle", memory, event, chains[event["chain_id"]], version,
                DummyScorer(), DummyController(),
            )
            self.assertTrue(decision["correct"])
        self.assertEqual([row["chain_id"] for row in memory], ["a0", "a1", "p0", "p1"])
        self.assertEqual([row["version"] for row in memory[-2:]], [1, 1])

    def test_clean_synthetic_aggregate_passes_gate(self):
        rows = []
        for group in ("g0", "g1"):
            for surface in ("exact", "rephrase"):
                for chain_kind in ("primary", "auxiliary"):
                    arms = {}
                    for arm in ARMS:
                        correct = int(arm in ("learned_auto", "oracle_auto", "oracle_oracle"))
                        arms[arm] = {
                            "candidate_correct": correct,
                            "answer_nll": .1 if correct else 2.,
                            "selection_correct": correct,
                            "stale_selection": 0,
                            "wrong_chain_selection": 0,
                            "defer": 1 - correct,
                            "memory_size": 4,
                            "generation": {"token_exact": correct},
                        }
                    rows.append({"group_id": group, "surface": surface,
                                 "chain_kind": chain_kind, "phase": "update_2",
                                 "final": True, "arms": arms})
        decisions = {
            policy: [
                {"correct": 1, "event_kind": kind}
                for kind in ("primary_initial", "hard_retain", "primary_update")
            ] for policy in STATE_POLICIES
        }
        states = [{"decisions": decisions}]
        result = aggregate(rows, states)
        self.assertTrue(result["gate"]["passed"])
        self.assertEqual(result["strongest_fixed_control"], "cosine_auto")
        self.assertEqual(result["metrics"]["learned_auto"]["candidate_accuracy"], 1.)


if __name__ == "__main__":
    unittest.main()
