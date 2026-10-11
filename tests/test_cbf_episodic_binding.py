import unittest

from tasks.train_cbf_episodic_binding import (
    ARMS, episode_schedule, episodic_gate, rotation_episodes,
)


class SpaceTokenizer:
    def __init__(self):
        self.values = {}

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        value = text.strip()
        return list(self.values[value])


def synthetic_rows():
    rows = []
    for context_index, context in enumerate(("c0", "c1")):
        values = [f"code_{context_index}{index}" for index in range(4)]
        patterns = [[100 + 10 * context_index + index] for index in range(4)]
        context_ids = [1, 2] + [token for pattern in patterns for token in pattern] + [9]
        for index, value in enumerate(values):
            rows.append({
                "id": f"{context}.q{index}", "context_id": context,
                "split": "train", "group_id": context,
                "answer": value, "answer_ids": [200 + index],
                "context_ids": context_ids, "fact_suffix_tokens": 5,
            })
    return rows


class EpisodicConstructionTests(unittest.TestCase):
    def test_rotations_change_every_query_target_and_context(self):
        rows = synthetic_rows()
        tokenizer = SpaceTokenizer()
        vocab = {}
        for row in rows:
            context_index = int(row["context_id"][-1])
            query_index = int(row["id"][-1])
            tokenizer.values[row["answer"]] = [100 + 10 * context_index + query_index]
            vocab[row["answer"]] = {"ids": [200 + query_index], "digits": [0], "prefix": []}
        episodes = rotation_episodes(rows, tokenizer, vocab)
        variants = episodes["c0"]
        self.assertEqual([item["targets"] for item in variants], [
            ["code_00", "code_01", "code_02", "code_03"],
            ["code_01", "code_02", "code_03", "code_00"],
            ["code_02", "code_03", "code_00", "code_01"],
            ["code_03", "code_00", "code_01", "code_02"],
        ])
        self.assertEqual(len({tuple(item["context_ids"]) for item in variants}), 4)

    def test_episodic_schedule_balances_all_four_assignments(self):
        rows = synthetic_rows()
        schedule = episode_schedule(rows, "episodic_assignment", 4, 7)
        for context in ("c0", "c1"):
            self.assertEqual(sorted(shift for item, shift in schedule if item == context), [0, 1, 2, 3])
        fixed = episode_schedule(rows, "fixed_assignment", 4, 7)
        self.assertTrue(all(shift == 0 for _, shift in fixed))
        self.assertEqual(len(fixed), len(schedule))


class EpisodicGateTests(unittest.TestCase):
    def test_gate_requires_causal_assignment_separation(self):
        comparisons = {"counterfactual": {"digit_gain_bootstrap_95ci": [.1, .2]}}
        dev = {
            "generation": {"full_kv": {"code_correct": .9}},
            "sets": {
                "context4": {
                    "policies": {"correct": {"top1": .75}, "counterfactual": {"top1": .25}},
                    "correct_comparisons": comparisons,
                },
                "global16": {"policies": {"correct": {"top1": .3}}},
            },
        }
        self.assertTrue(episodic_gate(dev)["passed"])
        dev["sets"]["context4"]["policies"]["counterfactual"]["top1"] = .6
        self.assertFalse(episodic_gate(dev)["passed"])

    def test_arm_names_are_fixed(self):
        self.assertEqual(ARMS, ("fixed_assignment", "episodic_assignment"))


if __name__ == "__main__":
    unittest.main()
