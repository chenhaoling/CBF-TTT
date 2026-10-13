import unittest

from tasks.build_long_session_sparse_corpus import build_episode


def fact(index, subject=None):
    return {"subject": subject or f"subject-{index}", "prompt": f"prompt {index}",
            "rephrase_prompt": f"rephrase {index}", "original": f"old {index}",
            "updated": f"new {index}", "locality_prompts": [f"local a {index}", f"local b {index}"],
            "source_row_sha256": f"hash-{index}", "source_session_id": f"session-{index}",
            "fact_index": index, "chain_id": f"hash-{index}"}


class SparseCorpusTest(unittest.TestCase):
    def test_episode_has_one_update_and_fixed_shape(self):
        facts = [fact(index) for index in range(20)]; table = {}
        row = build_episode(facts[0], facts, 16, "train", table)
        self.assertEqual(len(row["sessions"]), 16)
        self.assertTrue(all(len(session["units"]) == 4 for session in row["sessions"]))
        updates = [session for session in row["sessions"] if session["desired"] == "local"]
        self.assertEqual(len(updates), 1)
        old = next(session for session in row["sessions"] if session["event_kind"] == "primary_old")
        self.assertEqual(updates[0]["target_session_id"], old["session_id"])
        self.assertEqual(sum(session["event_kind"] == "hard_retain" for session in row["sessions"]), 2)

    def test_deterministic(self):
        facts = [fact(index) for index in range(8)]
        left, right = {}, {}
        self.assertEqual(build_episode(facts[0], facts, 32, "dev", left),
                         build_episode(facts[0], facts, 32, "dev", right))
        self.assertEqual(left, right)


if __name__ == "__main__": unittest.main()
