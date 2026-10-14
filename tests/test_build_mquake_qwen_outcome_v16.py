"""Tests for deterministic post-V15 MQuAKE outcome sampling."""

import unittest

from tasks.build_mquake_qwen_outcome_v16 import (
    FOLDS, LENGTHS, PILOT_PER_CELL, identity, select_formal, select_pilot,
)


def episode(subject, fold, length, index, relation_matched=True):
    prefix = f"{fold}-{length}-{index}"
    sessions = [
        {"session_id": f"{prefix}-d", "units": ["d"], "event_kind": "distractor",
         "desired": "retain", "target_session_id": None, "hard_relation_matched": None},
        {"session_id": f"{prefix}-o", "units": ["o"], "event_kind": "primary_old",
         "desired": "retain", "target_session_id": None, "hard_relation_matched": None},
        {"session_id": f"{prefix}-h", "units": ["h"], "event_kind": "hard_retain",
         "desired": "retain", "target_session_id": None,
         "hard_relation_matched": relation_matched},
        {"session_id": f"{prefix}-n", "units": ["n"], "event_kind": "primary_update",
         "desired": "local", "target_session_id": f"{prefix}-o", "hard_relation_matched": None},
    ]
    return {"episode_id": prefix, "primary_subject_hash": subject, "length": length,
            "split": "train", "sessions": sessions}


class OutcomeBuildTest(unittest.TestCase):
    def test_query_identity_uses_anchor_units(self):
        row = episode("s", 0, 16, 0)
        texts = {key: f"Memory statement: {key}?\nValue: value-{key}" for key in "dohn"}
        got = identity(row, 0, texts)
        self.assertEqual(got["old"]["unit_id"], "o")
        self.assertEqual(got["current"]["unit_id"], "n")
        self.assertEqual(got["query_step"], 3)
        self.assertEqual(got["hard_retain"]["unit_id"], "h")
        self.assertTrue(got["hard_relation_matched"])
        self.assertEqual(got["unrelated"]["unit_id"], "d")

    def test_identity_records_relation_fallback(self):
        row = episode("s", 0, 16, 0, relation_matched=False)
        texts = {key: f"Memory statement: {key}?\nValue: value-{key}" for key in "dohn"}
        got = identity(row, 0, texts)
        self.assertFalse(got["hard_relation_matched"])

    def test_safety_units_are_visible_at_query_step(self):
        row = episode("s", 0, 16, 0)
        row["sessions"].append({"session_id": "future", "units": ["f"],
                                "event_kind": "distractor", "desired": "retain",
                                "target_session_id": None, "hard_relation_matched": None})
        texts = {key: f"Memory statement: {key}?\nValue: value-{key}" for key in "dohnf"}
        got = identity(row, 0, texts)
        self.assertNotEqual(got["unrelated_session_id"], "future")

    def test_pilot_is_complete_and_deterministic(self):
        rows, folds = [], {}
        for fold in range(FOLDS):
            for length in LENGTHS:
                for index in range(3):
                    subject = f"s{fold}-{length}-{index}"
                    rows.append(episode(subject, fold, length, index)); folds[subject] = fold
        first = select_pilot(rows, folds)
        second = select_pilot(list(reversed(rows)), folds)
        self.assertEqual([row["episode_id"] for row in first],
                         [row["episode_id"] for row in second])
        self.assertEqual(len(first), FOLDS * len(LENGTHS) * PILOT_PER_CELL)

    def test_formal_has_one_row_per_subject_length(self):
        rows = []
        for subject in ("a", "b", "c"):
            for length in LENGTHS:
                rows.extend(episode(subject, 0, length, index) for index in range(3))
        selected = select_formal(rows)
        self.assertEqual(len(selected), 3 * len(LENGTHS))
        self.assertEqual(len({(row["primary_subject_hash"], row["length"])
                              for row in selected}), len(selected))


if __name__ == "__main__":
    unittest.main()
