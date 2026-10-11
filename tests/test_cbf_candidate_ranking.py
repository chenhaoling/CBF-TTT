"""Candidate-set, ranking, aggregate, and interpretation invariants."""

import copy
import unittest

from tasks.evaluate_cbf_candidate_ranking import (
    aggregate, build_candidate_sets, interpretation, ranking,
)


def rows():
    output = []
    for split in ("train", "dev"):
        for group_index in range(4):
            group = f"{split}{group_index}"
            shared = [f"code_{group_index}{index:04d}" for index in range(1, 4)]
            for twin in range(2):
                values = [f"code_{group_index}{twin}000"] + shared
                for question, answer in enumerate(values):
                    output.append({
                        "id": f"{group}.{twin}.q{question}", "group_id": group, "split": split,
                        "context_id": f"{group}.{twin}",
                        "twin_context_id": f"{group}.{1-twin}",
                        "wrong_context_id": f"{split}{(group_index+1)%4}.{twin}",
                        "twin_row_id": f"{group}.{1-twin}.q{question}",
                        "anchor_query": question == 0, "answer": answer,
                    })
    return output


class CandidateRankingTests(unittest.TestCase):
    def test_nested_candidate_sets_are_deterministic(self):
        data = rows()
        anchor = data[0]
        first = build_candidate_sets(data, anchor)
        second = build_candidate_sets(copy.deepcopy(data), copy.deepcopy(anchor))
        self.assertEqual(first, second)
        self.assertEqual({name: len(value) for name, value in first.items()},
                         {"pair2": 2, "world5": 5, "global16": 16})
        self.assertTrue(set(first["pair2"]).issubset(first["world5"]))
        self.assertTrue(set(first["world5"]).issubset(first["global16"]))
        nonanchor = build_candidate_sets(data, data[1])
        self.assertNotIn("pair2", nonanchor)

    def test_ranking_uses_digit_nll_and_positive_margin(self):
        candidates = ["code_00001", "code_00002", "code_00003"]
        scores = [{"answer": value, "digit_nll": float(index), "answer_nll": float(index + 1)}
                  for index, value in enumerate(candidates)]
        result = ranking(scores, candidates, candidates[0])
        self.assertEqual(result["top1"], 1)
        self.assertEqual(result["rank"], 1)
        self.assertEqual(result["mrr"], 1)
        self.assertEqual(result["margin"], 1)
        result = ranking(scores, candidates, candidates[2])
        self.assertEqual(result["top1"], 0)
        self.assertEqual(result["rank"], 3)
        self.assertLess(result["margin"], 0)

    def test_aggregate_and_fixed_interpretation(self):
        records = []
        for split in ("train_probe", "dev"):
            for group in ("g0", "g1"):
                for index in range(4):
                    candidate_sets = {"world5": [f"v{x}" for x in range(5)],
                                      "global16": [f"v{x}" for x in range(16)]}
                    if index == 0:
                        candidate_sets["pair2"] = ["v0", "v1"]
                    policies = {}
                    for policy in ("correct", "twin", "wrong", "empty"):
                        top1 = 1 if policy == "correct" else 0
                        rankings = {}
                        for name in candidate_sets:
                            rankings[name] = {"top1": top1, "mrr": 1. if top1 else .5,
                                              "rank": 1 if top1 else 2, "margin": 1. if top1 else -1.,
                                              "true_digit_nll": 1. if top1 else 2.,
                                              "true_answer_nll": 1.5 if top1 else 2.5,
                                              "prediction": "v0" if top1 else "v1"}
                        policies[policy] = {"rankings": rankings}
                    records.append({"evaluation_split": split, "group_id": group,
                                    "candidate_sets": candidate_sets, "policies": policies})
        result = aggregate(records)
        self.assertEqual(result["dev"]["sets"]["global16"]["policies"]["correct"]["top1"], 1)
        self.assertEqual(result["dev"]["sets"]["pair2"]["rows"], 2)
        self.assertEqual(interpretation(result["dev"])["label"], "strong_latent_binding")


if __name__ == "__main__":
    unittest.main()
