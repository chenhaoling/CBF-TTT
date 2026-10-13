import unittest

from tasks.build_mquake_temporal_sessions import build_episode, split_subjects


def case(index, subject=None, relation="P6"):
    subject = subject or f"city-{index}"
    return {"case_id": index,
            "requested_rewrite": [{"subject": subject, "relation_id": relation,
                                   "question": f"Who leads {subject}?",
                                   "prompt": "The leader of {} is",
                                   "target_true": {"str": f"old-{index}"},
                                   "target_new": {"str": f"new-{index}"}}],
            "single_hops": [{"question": f"Where is channel-{index}?",
                             "cloze": f"Channel-{index} is in", "answer": subject},
                            {"question": f"Who leads {subject}?",
                             "cloze": f"The leader of {subject} is", "answer": f"old-{index}"}],
            "new_single_hops": [{"question": f"Where is channel-{index}?",
                                 "cloze": f"Channel-{index} is in", "answer": subject},
                                {"question": f"Who leads {subject}?",
                                 "cloze": f"The leader of {subject} is", "answer": f"new-{index}"}],
            "orig": {"edit_triples_idx": [1]}}


class MquakeTemporalSessionsTest(unittest.TestCase):
    def test_split_keeps_subject_together(self):
        rows = [case(index, f"subject-{index // 3}") for index in range(45)]
        assignment, loads = split_subjects(rows)
        self.assertEqual(set(assignment), {f"subject-{index}" for index in range(15)})
        self.assertEqual(sum(loads.values()), len(rows))
        self.assertTrue(all(loads[name] for name in ("train", "tune", "dev")))

    def test_episode_has_temporal_update_and_hard_retains(self):
        rows = [case(index) for index in range(12)]; table = {}
        episode = build_episode(rows[0], rows, 16, "dev", table)
        self.assertEqual(len(episode["sessions"]), 16)
        self.assertTrue(all(len(row["units"]) == 4 for row in episode["sessions"]))
        old = next(row for row in episode["sessions"] if row["event_kind"] == "primary_old")
        update = next(row for row in episode["sessions"] if row["event_kind"] == "primary_update")
        self.assertEqual(update["target_session_id"], old["session_id"])
        self.assertEqual(update["desired"], "local")
        self.assertEqual(sum(row["event_kind"] == "hard_retain"
                             for row in episode["sessions"]), 2)


if __name__ == "__main__":
    unittest.main()
