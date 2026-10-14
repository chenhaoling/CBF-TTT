import unittest

from tasks.build_post_v19_natural_semantic_headroom import (
    FOLDS,
    FORMAL_PER_FOLD,
    PILOT_PER_FOLD,
    allocate,
    build_episode,
    build_states,
    normalized_surface,
)


def source_row(fold, index, relation=None, source=None):
    relation = relation or f"P{fold * 20 + index % 10}"
    source = source or ("mquake" if index % 11 == 0 else "counterfact")
    subject = f"Subject {fold} {index}"
    return {
        "masked_surfaces": ["[SUBJECT] old", "[SUBJECT] current", "[SUBJECT] query"],
        "outer_fold": fold,
        "relation_id": relation,
        "row_id": f"{source}:{fold}:{index}",
        "source": source,
        "source_row_sha256": f"{fold:01x}{index:063x}"[-64:],
        "source_split": "observed_development" if source == "counterfact" else "train",
        "subject": subject,
        "subject_normalized": subject.lower(),
        "surfaces": [f"{subject} was", f"Now {subject} is", f"Currently {subject} equals"],
        "values": [f"Old-{fold}-{index}", f"New-{fold}-{index}"],
    }


class NaturalSemanticHeadroomTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = [source_row(fold, index) for fold in range(FOLDS)
                    for index in range(120)]

    def test_allocation_is_balanced_disjoint_and_deterministic(self):
        first, second = allocate(self.rows), allocate(list(reversed(self.rows)))
        self.assertEqual(first, second)
        self.assertEqual(len(first["pilot"]), FOLDS * PILOT_PER_FOLD)
        self.assertEqual(len(first["formal"]), FOLDS * FORMAL_PER_FOLD)
        pilot = {row["source_row_sha256"] for row in first["pilot"]}
        formal = {row["source_row_sha256"] for row in first["formal"]}
        self.assertFalse(pilot & formal)
        self.assertFalse({row["subject_normalized"] for row in first["pilot"]}
                         & {row["subject_normalized"] for row in first["formal"]})

    def test_semantic_oracle_deletes_target_not_same_relation_donor(self):
        split = allocate(self.rows)["pilot"]
        episode = build_episode(split[0], split, "pilot", 0)
        state = build_states(episode)
        self.assertEqual(state["exact_removed"], [])
        self.assertTrue(state["semantic_removed"])
        removed = set(state["semantic_removed"])
        self.assertNotIn(episode["current_session_id"], removed)
        same_relation_other_subject = [row for row in episode["sessions"]
                                       if row["role"] == "same_relation_other_subject"]
        self.assertTrue(same_relation_other_subject)
        self.assertTrue(all(row["session_id"] not in removed
                            for row in same_relation_other_subject))

    def test_question_surface_is_rendered_as_memory_fact(self):
        split = allocate(self.rows)["pilot"]
        target = dict(split[0], surfaces=[f"Who leads {split[0]['subject']}?",
                                          f"The leader of {split[0]['subject']} is"])
        split = [target, *split[1:]]
        episode = build_episode(target, split, "pilot", 0)
        self.assertIn("recorded answer", episode["sessions"][0]["statement"])

    def test_normalized_surface_is_exact_not_semantic(self):
        self.assertEqual(normalized_surface("  Is   Located In? "), "is located in")
        self.assertNotEqual(normalized_surface("is located in"),
                            normalized_surface("can be found within"))


if __name__ == "__main__":
    unittest.main()
